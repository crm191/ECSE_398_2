"""Plug-in test for RFT60-HA01 USB sensor on COM5.
Install first: python -m pip install pyserial
Run in Spyder with F5. Stop with Ctrl+C or the Stop button.
Values are in sensor coordinates. No automatic zeroing is applied.
"""
import struct
import time

PORT = "COM3"
BAUD = 115200
MODEL = "RFT60-HA01"
DIVIDERS = {MODEL: (50, 2000)}

def command(cmd, parameter=0):
    data = bytes([cmd, parameter, 0, 0, 0, 0, 0, 0])
    return b'\x55' + data + bytes([sum(data) & 255, 170])

def signed(value, bits):
    return value - (1 << bits) if value & (1 << (bits - 1)) else value

class Parser:
    def __init__(self):
        self.buffer = bytearray()
        self.bad_frames = 0

    def feed(self, chunk):
        self.buffer.extend(chunk)
        packets = []
        while self.buffer:
            start = self.buffer.find(b'\x55')
            if start < 0:
                self.buffer.clear()
                break
            del self.buffer[:start]
            if len(self.buffer) < 19:
                break
            frame = self.buffer[:19]
            if frame[18] != 170 or frame[17] != sum(frame[1:17]) & 255:
                self.bad_frames += 1
                del self.buffer[0]
                continue
            packets.append(bytes(frame[1:17]))
            del self.buffer[:19]
        return packets

def decode(data, model):
    if len(data) != 16 or data[0] not in (10, 11):
        raise ValueError('Not an F/T response')
    raw = struct.unpack('>6h', data[1:13])
    df, dt = DIVIDERS[model]
    return [v / (df if i < 3 else dt) for i, v in enumerate(raw)], data[13]

class RFT:
    def __init__(self, port, baud, model):
        import serial
        self.serial = serial.Serial(port, baud, timeout=0.01, write_timeout=0.5)
        self.parser = Parser()
        self.model = model
        # Stop any prior stream, then discard residual input.
        self.serial.write(command(12))
        self.serial.flush()
        time.sleep(0.1)
        self.serial.reset_input_buffer()

    def read(self):
        start = time.perf_counter()
        self.serial.write(command(10))
        self.serial.flush()
        deadline = start + 0.5
        while time.perf_counter() < deadline:
            for data in self.parser.feed(self.serial.read(1)):
                if data[0] == 10:
                    end = time.perf_counter()
                    values, overload = decode(data, self.model)
                    return values, overload, start, end
        raise TimeoutError('No valid RFT response within 0.5 s; check interface, wiring and baud')

    def close(self):
        try:
            self.serial.write(command(12))
        finally:
            self.serial.close()

def main():
    sensor = None
    try:
        sensor = RFT(PORT, BAUD, MODEL)
        print(f"Connected to {MODEL} on {PORT}")
        print("Force: N | Torque: N*m | Press Ctrl+C to stop")
        while True:
            values, overload, _, _ = sensor.read()
            fx, fy, fz, tx, ty, tz = values
            print(
                f"Fx={fx:8.3f}  Fy={fy:8.3f}  Fz={fz:8.3f} N  |  "
                f"Tx={tx:8.4f}  Ty={ty:8.4f}  Tz={tz:8.4f} N*m"
            )
            if overload & 0x3F:
                axes = [name for bit, name in enumerate(
                    ("Tz", "Ty", "Tx", "Fz", "Fy", "Fx"))
                    if overload & (1 << bit)]
                print("OVERLOAD: " + ", ".join(axes))
                break
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Stopped.")
    except (OSError, TimeoutError) as exc:
        print(f"Cannot read sensor: {exc}")
        print("Check COM5, close other sensor programs, and verify sensor power/baud.")
    finally:
        if sensor is not None:
            sensor.close()

if __name__ == "__main__":
    main()
