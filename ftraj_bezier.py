from math import sin, cos, pi
import numpy as np

from IPython import embed

"""
T = 0.4	#period of 0.4s

dx = 0.04
dz = 0.06

def ftraj1(DeltaT, x0, z0):	#arguments : time, initial position x and z
		global T, dx, dz
		x = []
		z = []
		for t in DeltaT:
			if t >= T:
				t %= T
			x.append(x0-dx*np.cos(2*np.pi*t/T))
			if t <= T/2.:
				z.append(z0+dz*np.sin(2*np.pi*t/T))
			else:
				z.append(0)
		return np.matrix([x,z])
"""

from numpy import matrix
from curves import ( polynomial, bezier, cubic_hermite_spline, piecewise_polynomial_curve, piecewise_bezier_curve, piecewise_cubic_hermite_curve )

import matplotlib.pylab as plt

N_DISCRETIZATION = 100

Vdesired = 0.04		# desired velocity of the robot
L = 0.04			# desired leg stride in a gait cycle
Tst = L/Vdesired	# time of the stance phase
vD = 0.001			# virtual displacement to ensure the body stability

Tsw = 0.25			# time of the swing phase
T = Tst + Tsw		# time of one gait cycle
h = 0.03			# maximal height of the foot

# Bezier curves for the swing phase

P0  = [   0., 0. ]
P1  = [-0.005, 0. ]
P2  = [-0.02, 0.02]
P3  = [-0.02, 0.02]
P4  = [-0.02, 0.02]
P5  = [ 0.02, 0.02]
P6  = [ 0.02, 0.02]
P7  = [ 0.02, 0.03]
P8  = [ 0.06, 0.03]
P9  = [ 0.06, 0.03]
P10 = [ 0.045, 0. ]
P11 = [    L, 0. ]

waypoints = matrix([P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11]).T
bc = bezier(waypoints, 0.0, Tsw)

dt = np.linspace(0., Tsw, N_DISCRETIZATION)

# Discretize the bezier curves into 2 lists

bcX = []
bcZ = []

for i in dt: 
	bcX.append(bc(i)[0,0]) 
	bcZ.append(bc(i)[1,0])

def ftraj_swing(DeltaT, P0):
	x = []
	z = []
	for i,t in enumerate(DeltaT):
		t %= T
		if i < N_DISCRETIZATION :
			x.append(bcX[i])
			z.append(bcZ[i])
	return x,z
	
def ftraj_stance(DeltaT, P0):
	x = []
	z = []
	for i,t in enumerate(DeltaT):
		t %= T
		if t > Tsw:
			t -= Tsw
			x.append(P11[0] - Vdesired * t)
			z.append(-vD * cos(pi * (0.5 - Vdesired * t / L)) - P0[1])
	return x,z

def ftraj(DeltaT, P0):
	x = []
	z = []
	for i,t in enumerate(DeltaT):
		t %= T
		if i < N_DISCRETIZATION :
			x.append(bcX[i])
			z.append(bcZ[i])
		if t > Tsw:
			t -= Tsw
			x.append(P11[0] - Vdesired * t)
			z.append(-vD * cos(pi * (0.5 - Vdesired * t / L)) - P0[1])
	return x,z

N_SIMULATION = int(N_DISCRETIZATION * T / Tsw)

DeltaT = np.linspace(0., T, N_SIMULATION)

Xsw, Zsw = ftraj_swing(DeltaT, P0)
Xst, Zst = ftraj_stance(DeltaT, P0)

X,Z = ftraj(DeltaT, P0)

plt.figure(1)
plt.plot(Xsw, Zsw, 'g')
plt.plot(Xst, Zst, 'r')
plt.plot(P0[0], P0[1], 'x', label="control points")
plt.plot(P1[0], P1[1], 'x')
plt.plot(P2[0], P2[1], 'x')
plt.plot(P3[0], P3[1], 'x')
plt.plot(P4[0], P4[1], 'x')
plt.plot(P5[0], P5[1], 'x')
plt.plot(P6[0], P6[1], 'x')
plt.plot(P7[0], P7[1], 'x')
plt.plot(P8[0], P8[1], 'x')
plt.plot(P9[0], P9[1], 'x')
plt.plot(P10[0], P10[1], 'x')
plt.plot(P11[0], P11[1], 'x')
plt.xlabel('x (m)')
plt.ylabel('z (m)')
plt.legend()
plt.title("Foot trajectory using Bezier curve")

plt.figure(2)
plt.plot(X,Z)
plt.xlabel('x (m)')
plt.ylabel('z (m)')
plt.title("Foot trajectory using Bezier curve")
plt.show()

"""
DeltaT = np.linspace(0,2*T,300)
x0 = -0.19
z0 = 0
X = np.array(ftraj2(DeltaT, x0, z0)[0].T)
Z = np.array(ftraj2(DeltaT, x0, z0)[1].T)
plt.plot(X,Z)
plt.xlabel('displacement by x (m)')
plt.ylabel('displacement by z (m)')
plt.grid()
plt.title('Parametric curve of the trajectory')
plt.show()
"""
