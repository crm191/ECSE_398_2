"""
rft_sensor.py - Robotous RFT60-HA01 (USB) driver + logger + live plot.

  python rft_sensor.py --port COM4 --info          # connection test only
  python rft_sensor.py --port COM4                 # live plot + CSV, 200 Hz
  python rft_sensor.py --port COM4 --rate 100 --filter 50 --duration 60
"""
import argparse, csv, struct, threading, time
from collections import deque, namedtuple
from datetime import datetime

import numpy as np
import serial

SOP, EOP = 0x55, 0xAA
PKT_LEN = 19                      # SOP + 16 data + checksum + EOP

CMD_MODEL, CMD_SERIAL, CMD_FW = 1, 2, 3
CMD_SET_FILTER, CMD_READ_ONCE, CMD_START, CMD_STOP = 8, 10, 11, 12
CMD_SET_RATE, CMD_BIAS, CMD_OVERLOAD = 15, 17, 18

# model -> (DF, DT) from manual section 3.6.11
DIVIDERS = {"RFT80-6A02": (50, 1000), "RFT80-6A01": (50, 1000),
            "RFT64-6A01": (50, 1000), "RFT64-SB01": (50, 2000),
            "RFT60-HA01": (50, 2000), "RFT44-SB01": (50, 2000),
            "RFT40-SA01": (50, 2000)}
RATE_PARAM = {10: 1, 20: 2, 50: 3, 100: 4, 200: 5, 333: 6, 500: 7, 1000: 8}
MAX_RATE_AT_BAUD = {57600: 200, 115200: 333, 230400: 500,
                    460800: 500, 921600: 1000}
FILTER_PARAM = {500: 1, 300: 2, 200: 3, 150: 4, 100: 5, 50: 6, 40: 7,
                30: 8, 20: 9, 10: 10, 5: 11, 3: 12, 2: 13, 1: 14}

CHANNELS = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
Sample = namedtuple("Sample", "t ft overload")   # t = PC perf_counter [s]


class RFTSensor:
    def __init__(self, port="COM4", baud=115200):
        self.baud = baud
        self.ser = serial.Serial(port, baud, bytesize=8, parity="N",
                                 stopbits=1, timeout=0.05)
        self.df, self.dt = 50.0, 2000.0          # default = RFT60-HA01
        self.model = "?"
        self._buf = bytearray()
        self.samples = deque(maxlen=50000)       # thread-safe FIFO
        self._stop = threading.Event()
        self._thread = None
        # make sure the sensor is idle (it may still be streaming)
        self._send(CMD_STOP)
        time.sleep(0.1)
        self.ser.reset_input_buffer()
        self._buf.clear()

    # ---------------- low-level protocol ----------------
    @staticmethod
    def _packet(cmd, *params):
        data = bytearray(8)
        data[0] = cmd
        for i, p in enumerate(params):
            data[1 + i] = p
        return bytes([SOP]) + bytes(data) + bytes([sum(data) & 0xFF, EOP])

    def _send(self, cmd, *params):
        self.ser.write(self._packet(cmd, *params))

    def _extract(self):
        """Pull one valid 19-byte packet out of the buffer, resyncing."""
        b = self._buf
        while len(b) >= PKT_LEN:
            if (b[0] == SOP and b[PKT_LEN - 1] == EOP
                    and (sum(b[1:17]) & 0xFF) == b[17]):
                payload = bytes(b[1:17])
                del b[:PKT_LEN]
                return payload
            del b[0]          # 0x55 can appear inside data -> slide by 1
        return None

    def _read_packet(self, timeout=1.0):
        end = time.perf_counter() + timeout
        while time.perf_counter() < end:
            p = self._extract()
            if p is not None:
                return p
            self._buf.extend(self.ser.read(max(1, self.ser.in_waiting)))
        return None

    def _query(self, cmd, *params, timeout=1.0):
        """Send a command and wait for the response with the same ID."""
        self._send(cmd, *params)
        end = time.perf_counter() + timeout
        while time.perf_counter() < end:
            p = self._read_packet(end - time.perf_counter())
            if p is not None and p[0] == cmd:
                return p
        raise TimeoutError(f"No response to command {cmd}")

    # ---------------- sensor info / config ----------------
    def info(self):
        def txt(p):
            return p[1:16].split(b"\x00")[0].decode("ascii", "ignore").strip()
        self.model = txt(self._query(CMD_MODEL))
        serial_no = txt(self._query(CMD_SERIAL))
        fw = txt(self._query(CMD_FW))
        for name, (df, dt) in DIVIDERS.items():
            if name in self.model:
                self.df, self.dt = float(df), float(dt)
                break
        else:
            print(f"WARNING: unknown model '{self.model}', "
                  f"using DF={self.df}, DT={self.dt}")
        return self.model, serial_no, fw

    def configure(self, rate_hz=200, filter_hz=None):
        """Call while idle (before start). Settings persist in the sensor."""
        if rate_hz not in RATE_PARAM:
            raise ValueError(f"rate must be one of {sorted(RATE_PARAM)}")
        if rate_hz > MAX_RATE_AT_BAUD.get(self.baud, 200):
            raise ValueError(f"{rate_hz} Hz not allowed at {self.baud} baud")
        r = self._query(CMD_SET_RATE, RATE_PARAM[rate_hz])
        if r[1] != 1:
            raise RuntimeError(f"Set rate failed, error code {r[2]}")
        if filter_hz is None:
            r = self._query(CMD_SET_FILTER, 0, 0)          # no filter
        else:
            r = self._query(CMD_SET_FILTER, 1, FILTER_PARAM[filter_hz])
        if r[1] != 1:
            raise RuntimeError(f"Set filter failed, error code {r[2]}")

    def read_overload_counts(self):
        p = self._query(CMD_OVERLOAD)
        return dict(zip(CHANNELS, p[1:7]))

    # ---------------- streaming ----------------
    def start(self):
        self._send(CMD_START)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            p = self._read_packet(0.2)
            if p is None or p[0] not in (CMD_READ_ONCE, CMD_START):
                continue
            raw = np.array(struct.unpack(">6h", p[1:13]), dtype=float)
            ft = np.concatenate((raw[:3] / self.df, raw[3:] / self.dt))
            self.samples.append(Sample(time.perf_counter(), ft, p[13] & 0x3F))

    def set_bias(self, on=True):
        """Hardware tare. Only valid while streaming; not saved on reboot."""
        self._send(CMD_BIAS, 1 if on else 0)

    def drain(self):
        out = []
        while True:
            try:
                out.append(self.samples.popleft())
            except IndexError:
                return out

    def collect(self, n, timeout=10):
        out, t0 = [], time.perf_counter()
        while len(out) < n and time.perf_counter() - t0 < timeout:
            out.extend(self.drain())
            time.sleep(0.005)
        return out

    def close(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
        try:
            self._send(CMD_STOP)
            time.sleep(0.05)
        finally:
            self.ser.close()


# ====================== command-line app ======================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM4")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--rate", type=int, default=200, help="output rate [Hz]")
    ap.add_argument("--filter", type=int, default=None,
                    help="sensor low-pass cutoff [Hz] (default: off)")
    ap.add_argument("--tare", choices=["sw", "hw", "none"], default="sw")
    ap.add_argument("--tare-n", type=int, default=200)
    ap.add_argument("--duration", type=float, default=30)
    ap.add_argument("--window", type=int, default=1000, help="plot samples")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--info", action="store_true")
    args = ap.parse_args()

    s = RFTSensor(args.port, args.baud)
    try:
        model, sn, fw = s.info()
        print(f"Model: {model} | S/N: {sn} | Firmware: {fw}")
        print(f"DF={s.df:g}, DT={s.dt:g}")
        if args.info:
            print("Overload counts:", s.read_overload_counts())
            return

        s.configure(args.rate, args.filter)
        s.start()
        time.sleep(0.3)
        s.drain()

        offset = np.zeros(6)
        if args.tare == "hw":
            s.set_bias(True); time.sleep(0.2); s.drain()
            print("Hardware bias set.")
        elif args.tare == "sw":
            print(f"Taring ({args.tare_n} samples) - keep sensor UNLOADED...")
            got = s.collect(args.tare_n)
            if not got:
                raise RuntimeError("No data received. Check port/power.")
            arr = np.array([x.ft for x in got])
            offset = arr.mean(axis=0)
            print("Offset:", np.round(offset, 4))
            print("Noise std (N / N*m):", np.round(arr.std(axis=0), 4))

        fname = f"rft_{datetime.now():%Y%m%d_%H%M%S}.csv"
        f = open(fname, "w", newline="")
        w = csv.writer(f)
        w.writerow(["time_s"] + CHANNELS + ["overload_bits"])

        if not args.no_plot:
            import matplotlib.pyplot as plt
            plt.ion()
            fig, (axF, axT) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
            bt = deque(maxlen=args.window)
            bf = [deque(maxlen=args.window) for _ in range(6)]
            lines = [(axF if i < 3 else axT).plot([], [], label=c)[0]
                     for i, c in enumerate(CHANNELS)]
            axF.set_ylabel("Force [N]"); axT.set_ylabel("Torque [N·m]")
            axT.set_xlabel("Time [s]"); axF.legend(loc="upper left")
            axT.legend(loc="upper left")
            last_draw = 0

        t0, n = None, 0
        try:
            while True:
                for smp in s.drain():
                    if t0 is None:
                        t0 = smp.t
                    t = smp.t - t0
                    ft = smp.ft - offset
                    w.writerow([f"{t:.5f}"] + [f"{v:.5f}" for v in ft]
                               + [smp.overload])
                    n += 1
                    if smp.overload:
                        print(f"OVERLOAD bits {smp.overload:06b} (Fx..Tz)")
                    if not args.no_plot:
                        bt.append(t)
                        for i in range(6):
                            bf[i].append(ft[i])
                if t0 is not None and (time.perf_counter() - t0) > args.duration:
                    break
                if not args.no_plot:
                    if time.perf_counter() - last_draw > 0.05 and bt:
                        for i, ln in enumerate(lines):
                            ln.set_data(bt, bf[i])
                        for ax in (axF, axT):
                            ax.relim(); ax.autoscale_view()
                        plt.pause(0.001)
                        last_draw = time.perf_counter()
                        if not plt.fignum_exists(fig.number):
                            break
                else:
                    time.sleep(0.01)
        except KeyboardInterrupt:
            print("Stopped by user.")
        finally:
            f.close()
            if t0 is not None:
                print(f"{n} samples, ~{n / max(1e-9, (time.perf_counter() - t0)):.0f} Hz "
                      f"average -> {fname}")
    finally:
        s.close()


if __name__ == "__main__":
    main()