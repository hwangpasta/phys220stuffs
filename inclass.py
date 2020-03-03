import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""
data = np.loadtxt("circular.txt", float)
plt.imshow(data, origin="lower")
#plt.gray()
plt.colorbar()
plt.savefig("circular.pdf", dpi=600)
plt.show()
"""
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
"""
"""
#plot in 3D

delta = 0.1
x = np.arange(-3.0, 3.0, delta)
y = np.arange(-3.0, 3.0, delta)

X, Y = np.meshgrid(x, y)
Z = np.sin(X)*np.cos(Y)

fig = plt.figure()  # saves parameters from figure
ax = Axes3D(fig)

ax.plot_surface(X,Y,Z)
ax.plot_wireframe(X,Y,Z, color = 'r')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
"""


# plot in 3d scatter
def randrange(n, vmin, vmax):
    return (vmax - vmin) * np.random.rand(n) + vmin

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')  #set of plots together and takes a position 111

n = 100

for c, m, z1, zh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
    xs = randrange(n, 23, 32)
    ys = randrange(n, 0, 100)
    zs = randrange(n, z1, zh)
    ax.scatter(xs, ys, zs, marker=m)

ax.set_xlabel('X label')
ax.set_ylabel('Y label')
ax.set_zlabel('Z label')

plt.savefig("scatter3d.png", dpi=600)
plt.show()

# towards data science animations and whatnot