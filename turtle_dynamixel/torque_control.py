#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
#submodule = os.path.expanduser("~") + "/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/turtle_dynamixel"
#sys.path.append(submodule)
import numpy as np
from math import sqrt, sin, cos
from mass import mass

# Note: q0 is left, q3 is right. All joints are clockwise positive

def torque_control(q,dq,qd,dqd,ddqd, Kp, KD):

    M, C, G = mass(q, dq)
    # Changed C.dot(dq) to C.dot(dqd) -- Check if this is correct for the system dynamics
    tau = C.dot(dqd[:6]) + M.dot(ddqd[:6]) + Kp[:6,:6].dot((qd[:6]-q[:6])) + KD[:6,:6].dot(dqd[:6] - dq[:6]) + G.reshape(-1,1)
    # tau = Kp[:6,:6].dot((qd[:6]-q[:6])) + KD[:6,:6].dot(dqd[:6] - dq[:6])
        
    # print(f"[DEBUG] G: {G}\n")
    # tau[1] = -tau[1]
    
    return np.vstack((tau*150))
    # just to safely test the teacher traj
    # return np.vstack((tau,tau2))