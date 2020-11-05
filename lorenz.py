import numpy as np
import matplotlib.pyplot as plt

def lorenz(x, y, z,a=10, b=28, c=2.667):
    '''
    Dado un punto x, y, z y parámetros de atractor
    devuelve las derivadas respecto a las 3 componentes
    '''
    dx=a*(y-x)
    dy=x*(b-z)-y
    dz=x*y-c*z
    return dx, dy, dz

dt=0.01
pasos=10000

# vectores estado

xs=np.zeros(pasos+1)
ys=np.zeros(pasos+1)
zs=np.zeros(pasos+1)

# condiciones iniciales

xs[0], ys[0], zs[0] = (3., -45., -16)

for i in range(pasos):
    dx, dy, dz =lorenz(xs[i], ys[i], zs[i])
    xs[i+1]=xs[i]+(dt*dx)
    ys[i+1]=ys[i]+(dt*dy)
    zs[i+1]=zs[i]+(dt*dz)

# ploteamos
fig=plt.figure()

ax=fig.gca(projection='3d')

ax.plot(xs, ys, zs, lw=0.5)
plt.xlabel("x")
plt.ylabel("y")
plt.show()
