from mass import*
from torque_control import*
import numpy as np

def main():
    # Create test inputs for mass and torque control functions

    # Variables
    q1 = 360
    q2 = 330
    q3 = 330
    q4 = 360
    q5 = 360
    q6 = 360

    dq1 = 5
    dq2 = 0
    dq3 = 0
    dq4 = 5
    dq5 = 0
    dq6 = 0

    qd1 = 360
    qd2 = 360
    qd3 = 360
    qd4 = 360
    qd5 = 360
    qd6 = 360

    dqd1 = 0
    dqd2 = 0
    dqd3 = 0
    dqd4 = 0
    dqd5 = 0
    dqd6 = 0

    ddqd1 = 0
    ddqd2 = 0
    ddqd3 = 0
    ddqd4 = 0
    ddqd5 = 0
    ddqd6 = 0

    q = np.array([[q1], [q2], [q3], [q4], [q5], [q6]])  # Example joint positions (0-360 degrees)
    dq = np.array([[dq1], [dq2], [dq3], [dq4], [dq5], [dq6]])  # Example joint velocities
    
    qd = np.array([[qd1], [qd2], [qd3], [qd4], [qd5], [qd6]]) # Desired joint positions
    dqd = np.array([[dqd1], [dqd2], [dqd3], [dqd4], [dqd5], [dqd6]])    # Desired joint velocities
    ddqd = np.array([[ddqd1], [ddqd2], [ddqd3], [ddqd4], [ddqd5], [ddqd6]])    # Desired joint accelerations

    # Kp and Kd are 6x6 diag matrices.
    Kp = np.diag([2]*6)  # Proportional gain matrix
    Kd = np.diag([0.5]*6)  # Derivative gain matrix
    
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