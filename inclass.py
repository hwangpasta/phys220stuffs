import numpy as np
import matplotlib.pyplot as plt

"""
data = np.loadtxt("circular.txt", float)
plt.imshow(data, origin="lower")
#plt.gray()
plt.colorbar()
plt.savefig("circular.pdf", dpi=600)
plt.show()
"""

wavelength = 5.0 #cm
k = 2*np.pi/wavelength
ei0 = 1.0 # initial amplitude
separation = 20.0 #cm
side = 100.0 #cm
points = 500
spacing = side/points

x1 = side/2 + separation/2
y1 = side/2

x2 = side/2 - separation/2
y2 = side/2

# want to create a vector to add all the amplitudes when they are calculated

ei = np.empty([points,points], float) # vector of 500 x 500

# a loop to go ahead and integrate that

for i in range(points):
    y = spacing*i
    for j in range(points):
        x = spacing*j
        r1 = np.sqrt((x-x1)**2+(y-y1)**2)
        r2 = np.sqrt((x-x2)**2+(y-y2)**2)
        ei[i, j] = ei0*np.sin(k*r1) + ei0*np.sin(k*r2)

plt.imshow(ei, origin="lower",extent=[0,side, 0, side])
plt.savefig("interference.pdf", dpi=600)
plt.show()
