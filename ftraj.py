from math import sin, cos
import numpy as np
from IPython import embed

T = 0.4	#period of 2s

dx = 0.02
dz = 0.05

def ftraj(t, x0, z0):	#arguments : time, initial position x and z
	global T, d
	x = []
	z = []
	for dt in t:
		if dt >= T:
			dt %= T
		x.append(x0-dx*cos(2*np.pi*dt/T))
		if dt <= T/2:
			z.append(z0+dz*sin(2*np.pi*dt/T))
		else:
			z.append(0)
	return np.matrix([x,z])
	

def ftraj2(DeltaT, x0, z0):	#arguments : time, initial position x and z
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
		
		
import matplotlib.pylab as plt

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


embed()
