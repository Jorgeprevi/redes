import numpy as np
import matplotlib.pyplot as plt

def rossler(x, y, z,a=0.1, b=0.1, c=14):
    '''
    Dado un punto x, y, z y parámetros de atractor
    devuelve las derivadas respecto a las 3 componentes
    '''
    dx=-z-y
    dy=x+a*y
    dz=b+z*(x-c)
    return dx, dy, dz

dt=0.01
pasos=50000

# vectores estado

xs=np.empty(pasos+1)
ys=np.empty(pasos+1)
zs=np.empty(pasos+1)

# condiciones iniciales

xs[0], ys[0], zs[0] = (-9., 0., 0.)

for i in range(pasos):
    dx, dy, dz =rossler(xs[i], ys[i], zs[i])
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