#!/usr/bin/env python
import numpy as np
from numpy import cross, append, array
from scipy.linalg import expm
from math import pi
import math

"""
Use 'expm' for matrix exponential.
Angles are in radian, distance are in meters.
"""
BASE_X = -150
BASE_Y = 150
BASE_Z = 10
L1 = 152
L2 = 120
L3 = 244
L4 = -93
L5 = 213
L6 = 104
L7 = 85
L8 = 92
def Get_MS()->tuple:
	# =================== Your code starts here ====================#
	# Fill in the correct values for S1~6, as well as the M matrix
	M = np.eye(4)
	S = np.zeros((6,6))
	w0 = array([0,0,1])
	q0 = array([BASE_X,BASE_Y,BASE_Z+L1])
	S[0] = append(w0, cross(-w0,q0)) 
	w1 = array([0,1,0])
	q1 = q0 + array([0,L2,0])
	S[1] = append(w1,cross(-w1,q1)) 
	w2 = array([0,1,0])
	q2 = q1 + array([L3,0,0])
	S[2] = append(w2,cross(-w2,q2)) 
	w3 = array([0,1,0])
	q3 = q2 + array([L5,L4,0])
	S[3] = append(w3,cross(-w3,q3)) 
	w4 = array([1,0,0])
	q4 = q3 + array([0,L6,0])
	S[4] = append(w4,cross(-w4,q4)) 
	w5 = array([0,1,0])
	q5 = q4 + array([L7,0,0	])
	S[5] = append(w5, cross(-w5,q5)) 
	M[:3,-1] = (q5 + array([0,L8+59,53.5])).reshape(1,3)
	M[:3,:3] = array([[0,-1,0],[0,0,-1],[1,0,0]])

	# ==============================================================#
	return M, S

def w(w):
	return np.array([[0,-w[2],w[1]], [w[2],0,-w[0]], [-w[1], w[0],0]])

def S(s):
	a = np.eye(4)
	w_box = w(s[:3])
	a[:3,:3] = w_box
	a[:3,-1] = s[3:]
	a[3][3] = 0
	return a
"""
Function that calculates encoder numbers for each motor
"""
def lab_fk(theta1, theta2, theta3, theta4, theta5, theta6):

	# Initialize the return_value
	return_value = [None, None, None, None, None, None]

	# =========== Implement joint angle to encoder expressions here ===========
	print("Foward kinematics calculated:\n")

	# =================== Your code starts here ====================#

	T = np.eye(4)
	m,s = Get_MS()
	T = expm(S(s[0])*theta1) @ expm(S(s[1])*theta2) @ expm(S(s[2])*theta3) @ expm(S(s[3])*theta4) @ expm(S(s[4])*theta5) @ expm(S(s[5])*theta6) @ m
	# ==============================================================#

	print(str(T) + "\n")

	return_value[0] = theta1 + pi
	return_value[1] = theta2
	return_value[2] = theta3
	return_value[3] = theta4 - (0.5*pi)
	return_value[4] = theta5
	return_value[5] = theta6

	return return_value


"""
Function that calculates an elbow up Inverse Kinematic solution for the UR3
"""
def lab_invk(xWgrip, yWgrip, zWgrip, yaw_WgripDegree):
	# =================== Your code starts here ====================#
	
	theta1 = 0.0
	theta2 = 0.0
	theta3 = 0.0
	theta4 = 0.0
	theta5 = 0.0
	theta6 = 0.0
	
	# ==============================================================#
	return lab_fk(theta1, theta2, theta3, theta4, theta5, theta6)

"""
Z1 = 6.7
X1 = 17.8
Y1 = 32.2

Z2 = 28.6
Y2 = -13.2
X1 = 19.5

"""
