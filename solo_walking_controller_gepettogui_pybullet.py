# coding: utf8

import pybullet as p # PyBullet simulator
import time # Time module to sleep()
import pybullet_data
import pinocchio as pin       # Pinocchio library
import numpy as np # Numpy library
import robots_loader # Functions to load the SOLO quadruped
from pinocchio.utils import * # Utilitary functions from Pinocchio
from pinocchio.robot_wrapper import RobotWrapper # Robot Wrapper to load an URDF in Pinocchio
from os.path import dirname, exists, join # OS function for path manipulation

########################################################################

from pinocchio import Quaternion
from pinocchio.rpy import matrixToRpy
from time import sleep, time
from IPython import embed
from numpy.linalg import pinv
from math import sin, cos

### Spacemouse configuration
import spacenav as sp
import atexit

# function defining the feet's trajectory

def ftraj(t, x0, z0):	#arguments : time, initial position x and z
	global T, dx, dz
	x = []
	z = []
	if t >= T:
		t %= T
	x.append(x0-dx*cos(2*np.pi*t/T))
	if t <= T/2.:
		z.append(z0+dz*sin(2*np.pi*t/T))
	else:
		z.append(0)
	return np.matrix([x,z])
	
########################################################################

v_prev = np.zeros((14,1)) # velocity during the previous time step, of size (robot.nv,1)


## Function called from the main loop of the simulation ##
def callback_torques():
	global v_prev, solo, stop, q, qdes, t
	
	t_start = time()
	
	jointStates = p.getJointStates(robotId, revoluteJointIndices) # State of all joints
	baseState   = p.getBasePositionAndOrientation(robotId)
	baseVel = p.getBaseVelocity(robotId)

	# Info about contact points with the ground
	contactPoints_FL = p.getContactPoints(robotId, planeId, linkIndexA=2)  # Front left  foot 
	contactPoints_FR = p.getContactPoints(robotId, planeId, linkIndexA=5)  # Front right foot 
	contactPoints_HL = p.getContactPoints(robotId, planeId, linkIndexA=8)  # Hind  left  foot 
	contactPoints_HR = p.getContactPoints(robotId, planeId, linkIndexA=11) # Hind  right foot 

	# Sort contacts points to get only one contact per foot
	contactPoints = []
	contactPoints.append(getContactPoint(contactPoints_FL))
	contactPoints.append(getContactPoint(contactPoints_FR))
	contactPoints.append(getContactPoint(contactPoints_HL))
	contactPoints.append(getContactPoint(contactPoints_HR))

	# Joint vector for Pinocchio
	q = np.vstack((np.array([baseState[0]]).transpose(), np.array([baseState[1]]).transpose(), np.array([[jointStates[i_joint][0] for i_joint in range(len(jointStates))]]).transpose()))
	v = np.vstack((np.array([baseVel[0]]).transpose(), np.array([baseVel[1]]).transpose(), np.array([[jointStates[i_joint][1] for i_joint in range(len(jointStates))]]).transpose()))
	v_dot = (v-v_prev)/dt
	v_prev = v.copy()

########################################################################
 
	### Stack of Task : walking
	#compute/update all the joints and frames
	pin.forwardKinematics(solo.model, solo.data, qdes)
	pin.updateFramePlacements(solo.model, solo.data)
	
	# Getting the current height (on axis z) and the x-coordinate of the front left foot
	xz_FL = solo.data.oMf[ID_FL].translation[0::2]
	xz_FR = solo.data.oMf[ID_FR].translation[0::2]
	xz_HL = solo.data.oMf[ID_HL].translation[0::2]
	xz_HR = solo.data.oMf[ID_HR].translation[0::2]
	
	# Desired foot trajectory
	t += dt
	xzdes_FL = ftraj(t, xF0, zF0)
	xzdes_HR = ftraj(t, xH0, zH0)
	xzdes_FR = ftraj(t+T/2, xF0, zF0)
	xzdes_HL = ftraj(t+T/2, xH0, zH0)
	
	# Calculating the error
	err_FL = xz_FL - xzdes_FL
	err_FR = xz_FR - xzdes_FR
	err_HL = xz_HL - xzdes_HL
	err_HR = xz_HR - xzdes_HR
	
	# Computing the local Jacobian into the global frame
	oR_FL = solo.data.oMf[ID_FL].rotation
	oR_FR = solo.data.oMf[ID_FR].rotation
	oR_HL = solo.data.oMf[ID_HL].rotation
	oR_HR = solo.data.oMf[ID_HR].rotation
	
	# Getting the different Jacobians
	fJ_FL3 = pin.frameJacobian(solo.model, solo.data, qdes, ID_FL)[:3,-8:]	#Take only the translation terms
	oJ_FL3 = oR_FL*fJ_FL3	#Transformation from local frame to world frame
	oJ_FLxz = oJ_FL3[0::2,-8:]	#Take the x and z components
	
	fJ_FR3 = pin.frameJacobian(solo.model, solo.data, qdes, ID_FR)[:3,-8:]
	oJ_FR3 = oR_FR*fJ_FR3
	oJ_FRxz = oJ_FR3[0::2,-8:]
	
	fJ_HL3 = pin.frameJacobian(solo.model, solo.data, qdes, ID_HL)[:3,-8:]
	oJ_HL3 = oR_HL*fJ_HL3
	oJ_HLxz = oJ_HL3[0::2,-8:]
	
	fJ_HR3 = pin.frameJacobian(solo.model, solo.data, qdes, ID_HR)[:3,-8:]
	oJ_HR3 = oR_HR*fJ_HR3
	oJ_HRxz = oJ_HR3[0::2,-8:]
	
	# Displacement error
	nu = np.vstack([err_FL, err_FR, err_HL, err_HR])
	
	# Making a single x&z-rows Jacobian vector 
	J = np.vstack([oJ_FLxz, oJ_FRxz, oJ_HLxz, oJ_HRxz])
	
	# Computing the velocity
	vq_act = -Kp*pinv(J)*nu
	vq = np.concatenate((np.zeros([6,1]) , vq_act))

	# Computing the updated configuration
	qdes = pin.integrate(solo.model, qdes, vq * dt)
	
	# Update display in Gepetto-gui
	solo.display(qdes)
	
	#hist_err.append(np.linalg.norm(nu))
	hist_traj_x_FL.append(float(xzdes_FL[0]))
	hist_traj_x_FR.append(float(xzdes_FR[0]))
	hist_traj_x_HL.append(float(xzdes_HL[0]))
	hist_traj_x_HR.append(float(xzdes_HR[0]))
	hist_traj_Z1.append(float(xzdes_FL[1]))
	hist_traj_Z2.append(float(xzdes_FR[1]))
	
	hist_pos_x_FL.append(float(xz_FL[0]))
	hist_pos_z_FL.append(float(xz_FL[1]))
	hist_pos_x_FR.append(float(xz_FR[0]))
	hist_pos_z_FR.append(float(xz_FR[1]))
	hist_pos_x_HL.append(float(xz_HL[0]))
	hist_pos_z_HL.append(float(xz_HL[1]))
	hist_pos_x_HR.append(float(xz_HR[0]))
	hist_pos_z_HR.append(float(xz_HR[1]))
  
########################################################################

	# PD Torque controller
	Kp_PD = 10.
	Kd_PD = 0.3
	
	torques = Kp_PD * (qdes[7:] - q[7:]) + Kd_PD * (vq_act - v[6:])
	
	# Saturation to limit the maximal torque
	t_max = 3
	torques = np.maximum(np.minimum(torques, t_max * np.ones((8,1))), -t_max * np.ones((8,1)))

########################################################################
	"""
	# Control loop of 1/dt Hz
	while (time()-t_start)<dt:
		sleep(10e-6)
"""
########################################################################
		
	return torques


## Sort contacts points to get only one contact per foot ##
def getContactPoint(contactPoints):
	for i in range(0,len(contactPoints)):
		# There may be several contact points for each foot but only one of them as a non zero normal force
		if (contactPoints[i][9] != 0): 
			return contactPoints[i]
	return 0 # If it returns 0 then it means there is no contact point with a non zero normal force (should not happen) 

########################################################################
########################### START OF SCRIPT ############################
########################################################################

# Load the robot for Pinocchio
solo = robots_loader.loadSolo(True)
solo.initDisplay(loadModel=True)

########################################################################

q = solo.q0.copy()

qdes = np.zeros((15,1))

t = 0
dt = 0.001    # Integration step

# Convergence gain
Kp = 100.

# Getting the frame index of each foot
ID_FL = solo.model.getFrameId("FL_FOOT")
ID_FR = solo.model.getFrameId("FR_FOOT")
ID_HL = solo.model.getFrameId("HL_FOOT")
ID_HR = solo.model.getFrameId("HR_FOOT")

# Initial condition for the feet trajectory
# Front feet
xF0 = 0.19 	
zF0 = 0
# Hind feet
xH0 = -0.19 	
zH0 = 0
# Feet trajectory parameters
T = .5	#period of 0.5s
dx = 0.03
dz = 0.05

hist_err = []     #history of the error

hist_traj_x_FL = []
hist_traj_x_FR = []
hist_traj_x_HL = []
hist_traj_x_HR = []
hist_traj_Z1 = []
hist_traj_Z2 = []
hist_pos_x_FL = []
hist_pos_x_FR = []
hist_pos_x_HL = []
hist_pos_x_HR = []
hist_pos_z_FL = []
hist_pos_z_FR = []
hist_pos_z_HL = []
hist_pos_z_HR = []

########################################################################

# Start the client for PyBullet
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version

# Set gravity (disabled by default)
p.setGravity(0,0,-9.81)

# Load horizontal plane
p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = p.loadURDF("plane.urdf")

# Load Quadruped robot
robotStartPos = [0,0,0.35]
robotStartOrientation = p.getQuaternionFromEuler([0,0,0])
p.setAdditionalSearchPath("/opt/openrobots/share/example-robot-data/robots/solo_description/robots")
robotId = p.loadURDF("solo.urdf",robotStartPos, robotStartOrientation)
p.setTimeStep(dt)	# set the simulation time

# Disable default motor control for revolute joints
revoluteJointIndices = [0,1,3,4,6,7,9,10]
p.setJointMotorControlArray(robotId, 
							jointIndices= revoluteJointIndices, 
							 controlMode= p.VELOCITY_CONTROL,
					   targetVelocities = [0.0 for m in revoluteJointIndices],
								 forces = [0.0 for m in revoluteJointIndices])

# Enable torque control for revolute joints
jointTorques = [0.0 for m in revoluteJointIndices]
p.setJointMotorControlArray(robotId, revoluteJointIndices, controlMode=p.TORQUE_CONTROL, forces=jointTorques)

p.setTimeStep(dt)

# Launch the simulation
for i in range(1000):
   
	# Compute one step of simulation
	p.stepSimulation()
	
	# Callback Pinocchio to get joint torques
	jointTorques = callback_torques()

	# Set control torque for all joints
	p.setJointMotorControlArray(robotId, revoluteJointIndices, controlMode=p.TORQUE_CONTROL, forces=jointTorques)


import matplotlib.pyplot as plt

"""
plt.figure(1)
plt.subplot(211)
plt.plot(hist_traj_x_FL, 'b', label='reference')
plt.plot(hist_pos_x_FL, 'g', label='reality')
plt.title('Trajectory by x of the front left foot')
plt.subplot(212)
plt.plot(hist_traj_Z1, 'b', label='reference')
plt.plot(hist_pos_z_FL, 'g', label='reality')
plt.title('Trajectory by z of the front left foot')
plt.legend()

plt.figure(2)
plt.subplot(211)
plt.plot(hist_traj_x_FR, 'b', label='reference')
plt.plot(hist_pos_x_FR, 'g', label='reality')
plt.title('Trajectory by x of the front right foot')
plt.subplot(212)
plt.plot(hist_traj_Z2, 'b', label='reference')
plt.plot(hist_pos_z_FR, 'g', label='reality')
plt.title('Trajectory by z of the front right foot')
plt.legend()

plt.figure(3)
plt.subplot(211)
plt.plot(hist_traj_x_HR, 'b', label='reference')
plt.plot(hist_pos_x_HR, 'g', label='reality')
plt.title('Trajectory by x of the hind right foot')
plt.subplot(212)
plt.plot(hist_traj_Z1, 'b', label='reference')
plt.plot(hist_pos_z_HR, 'g', label='reality')
plt.title('Trajectory by z of the hind right foot')
plt.legend()

plt.figure(4)
plt.subplot(211)
plt.plot(hist_traj_x_HL, 'b', label='reference')
plt.plot(hist_pos_x_HL, 'g', label='reality')
plt.title('Trajectory by x of the hind left foot')
plt.subplot(212)
plt.plot(hist_traj_Z2, 'b', label='reference')
plt.plot(hist_pos_z_HL, 'g', label='reality')
plt.title('Trajectory by z of the hind left foot')
plt.legend()

plt.show()"""

embed()

# Shut down the client
p.disconnect()
