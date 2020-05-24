import numpy as np
import matplotlib.pyplot as plt

dt=0.001
pasos=30000

def vot(m, v=-0.5, a=1.01):
    dm=(2.0**(-a-1.0))*(1.0-m*m)*((1.0+v)*(1.0+m)**(a-1.0)-(1-v)*(1.0-m)**(a-1.0))
    return dm

m=np.zeros(pasos+1, dtype=float)
t=np.linspace(0, pasos*dt, pasos+1)
for k in range(-20,21):
    m[0]=float(k)*0.05
    for i in range(pasos):
        a=m[i]
        m[i+1]=m[i]+dt*vot(a)
    plt.plot(m)

plt.xlim(0,15000)
plt.xlabel("tiempo (a.u.)")
plt.ylabel("m")
plt.show()