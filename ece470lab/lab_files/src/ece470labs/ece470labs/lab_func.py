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
L9 = 53.5  # Horizontal suction-cup offset from joint 6 (mm).
L10 = 59  # Vertical tool extension below link 8 (mm).
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
	M[:3,-1] = (q5 + array([0,L8+L10,L9])).reshape(1,3)
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
def forward_kinematics(theta1, theta2, theta3, theta4, theta5, theta6):
    """Return the world-frame tool transform; translation is in millimeters.

    Inputs are geometric joint angles in radians, before encoder offsets.
    The existing Lab 3 product-of-exponentials model is kept unchanged.
    """
    angles = np.asarray([theta1, theta2, theta3, theta4, theta5, theta6],
                        dtype=float)
    if not np.all(np.isfinite(angles)):
        raise ValueError("Joint angles must be finite.")
    home, screws = Get_MS()
    transform = np.eye(4)
    for screw, angle in zip(screws, angles):
        transform = transform @ expm(S(screw) * angle)
    return transform @ home


def lab_fk(theta1, theta2, theta3, theta4, theta5, theta6):
    """Print Lab 3 FK and return the six ROS encoder angles in radians."""
    transform = forward_kinematics(theta1, theta2, theta3, theta4, theta5, theta6)
    print("Forward kinematics (world frame; matrix translation in mm):")
    print(transform)
    print("World gripper position (m):", transform[:3, 3] / 1000.0)
    return [theta1 + pi, theta2, theta3, theta4 - pi / 2, theta5, theta6]


def _unit_interval(value, description):
    """Reject unreachable geometry, allowing roundoff at workspace limits."""
    if value < -1.0 - 1e-10 or value > 1.0 + 1e-10:
        raise ValueError("Unreachable pose: " + description)
    return float(np.clip(value, -1.0, 1.0))


def lab_ik_angles(xWgrip, yWgrip, zWgrip, yaw_WgripDegree):
    """Solve the manual's elbow-up branch, returning geometric radians.

    World coordinates are in meters and yaw is in degrees about world +z.
    Link dimensions and the world-to-base translation come from Lab 3.
    See lab4/Lab4_Report.html for the illustrated derivation.
    """
    pose = np.asarray([xWgrip, yWgrip, zWgrip, yaw_WgripDegree], dtype=float)
    if not np.all(np.isfinite(pose)):
        raise ValueError("Position and yaw must be finite numbers.")
    yaw = math.radians(float(pose[3]) % 360.0)

    # 1. World -> base: aligned axes, base origin (-.150, .150, .010) m.
    xgrip = float(pose[0]) - BASE_X / 1000.0
    ygrip = float(pose[1]) - BASE_Y / 1000.0
    zgrip = float(pose[2]) - BASE_Z / 1000.0

    # 2. Joint-6 axis projected to the height of the suction cup.
    xcen = xgrip - L9 / 1000.0 * math.cos(yaw)
    ycen = ygrip - L9 / 1000.0 * math.sin(yaw)
    zcen = zgrip

    # 3. Top view: c = (r3 + L7) e_r + d e_t, d = L2 + L4 + L6.
    offset = (L2 + L4 + L6) / 1000.0
    radius = math.hypot(xcen, ycen)
    if radius < offset - 1e-12:
        raise ValueError("Unreachable pose: wrist center is inside the "
                         f"{offset:.3f} m waist-offset cylinder.")
    theta1 = math.atan2(ycen, xcen) - math.asin(
        _unit_interval(offset / radius, "waist offset exceeds wrist radius"))

    # 4. At theta6=0 the plate points along +e_t.
    theta6 = (theta1 + pi / 2 - yaw + pi) % (2 * pi) - pi

    # 5. Remove the horizontal wrist links and lift to the elbow plane.
    x3end = xcen - L7 / 1000.0 * math.cos(theta1) + offset * math.sin(theta1)
    y3end = ycen - L7 / 1000.0 * math.sin(theta1) - offset * math.cos(theta1)
    z3end = zcen + (L8 + L10) / 1000.0
    radial = x3end * math.cos(theta1) + y3end * math.sin(theta1)
    height = z3end - L1 / 1000.0
    distance = math.hypot(radial, height)
    upper_arm, forearm = L3 / 1000.0, L5 / 1000.0
    if (distance < abs(upper_arm - forearm) - 1e-12 or
            distance > upper_arm + forearm + 1e-12):
        raise ValueError("Unreachable pose: shoulder-to-wrist distance "
                         f"{distance:.6f} m is outside "
                         f"[{abs(upper_arm - forearm):.3f}, "
                         f"{upper_arm + forearm:.3f}] m.")

    # 6. Law of cosines; the shoulder-above-target branch is elbow-up.
    theta3 = math.acos(_unit_interval(
        (distance**2 - upper_arm**2 - forearm**2) /
        (2 * upper_arm * forearm), "elbow triangle cannot close"))
    # Equivalent to the shoulder cosine rule, but stays consistent with
    # theta3 at straight/folded boundaries despite floating-point roundoff.
    alpha = math.atan2(forearm * math.sin(theta3),
                       upper_arm + forearm * math.cos(theta3))
    theta2 = -math.atan2(height, radial) - alpha
    theta4 = -theta2 - theta3
    theta5 = -pi / 2
    return [theta1, theta2, theta3, theta4, theta5, theta6]


def lab_invk(xWgrip, yWgrip, zWgrip, yaw_WgripDegree):
    """Solve IK, verify using Lab 3 FK, and return ROS encoder radians."""
    angles = lab_ik_angles(xWgrip, yWgrip, zWgrip, yaw_WgripDegree)
    transform = forward_kinematics(*angles)
    target = np.asarray([xWgrip, yWgrip, zWgrip], dtype=float)
    residual = np.linalg.norm(transform[:3, 3] / 1000.0 - target)
    yaw = math.radians(float(yaw_WgripDegree) % 360.0)
    expected_rotation = np.array([[math.cos(yaw), -math.sin(yaw), 0.0],
                                  [math.sin(yaw), math.cos(yaw), 0.0],
                                  [0.0, 0.0, 1.0]])
    if residual > 1e-8 or not np.allclose(
            transform[:3, :3], expected_rotation, atol=1e-8, rtol=0):
        raise ValueError("IK failed the forward-kinematics consistency check.")
    print(f"IK/FK position residual: {residual:.3e} m")
    # The Lab 3 function applies theta1+pi and theta4-pi/2 exactly once.
    return lab_fk(*angles)


"""
Z1 = 6.7
X1 = 17.8
Y1 = 32.2

Z2 = 28.6
Y2 = -13.2
X1 = 19.5

"""
