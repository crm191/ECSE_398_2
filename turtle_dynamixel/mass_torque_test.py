from mass import*
from torque_control import*
import numpy as np

def main():
    # Create test inputs for mass and torque control functions
    q = np.array([120, 350, 270])  # Example joint positions (0-360 degrees)
    print(q[0][0])
    dq = np.array([5, 10, 15])     # Example joint velocities

    qd = np.array([130, 360, 280]) # Desired joint positions
    dqd = np.array([6, 12, 18])    # Desired joint velocities
    ddqd = np.array([1, 0, 0])    # Desired joint accelerations

    Kp = 2
    Kd = 0.5

    # Run mass test and see output
    print("Running mass test...")
    M, C, G = mass(q, dq)
    print("Mass matrix M:\n", M)
    print("Coriolis matrix C:\n", C)
    print("Gravity vector G:\n", G)

    # Run torque control test and see output
    print("\nRunning torque control test...")
    vstack = torque_control(q, dq, qd, dqd, ddqd, Kp, Kd)
    print("Torque control output:\n", vstack)

if __name__ == "__main__":
    main()