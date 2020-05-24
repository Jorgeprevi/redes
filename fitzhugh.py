import numpy as np
import matplotlib.pyplot as plt

dt=0.01
pasos=200000
#intens=0.5
a=0.7
b=0.8
phi=12.5
tiempo=np.linspace(0, pasos*dt, pasos+1)

V=np.zeros(pasos+1)
U=np.zeros(pasos+1)
inte=np.zeros(pasos+1)
for i in range(pasos):
    inte[i]=0.00001*float(i)
    V[i+1]=V[i]+dt*(V[i]-(V[i]**3)/3-U[i]+inte[i])
    U[i+1]=U[i]+dt*(V[i]+a-b*U[i])/phi


# ploteo V
plt.xlabel("T")
plt.ylabel("V")
plt.plot(tiempo,V)
plt.xlim(100,1900)
plt.show()

# ploteo I
plt.xlabel("T")
plt.ylabel("I")
plt.plot(tiempo,inte)
plt.xlim(100,1900)
plt.show()

# ploteo fase
plt.xlabel("V")
plt.ylabel("U")
plt.plot(V,U)
plt.show()

