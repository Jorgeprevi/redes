import numpy as np
import matplotlib.pyplot as plt
 
dt=0.01
pasos=101000


tiempo=np.linspace(0, pasos*dt, pasos+1)

# vectores estado
m=np.zeros(pasos+1, dtype=float)
h=np.zeros(pasos+1, dtype=float)
n=np.zeros(pasos+1, dtype=float)

vtaum=np.zeros(pasos+1, dtype=float)
vtauh=np.zeros(pasos+1, dtype=float)
vtaun=np.zeros(pasos+1, dtype=float)

vminf=np.zeros(pasos+1, dtype=float)
vhinf=np.zeros(pasos+1, dtype=float)
vninf=np.zeros(pasos+1, dtype=float)

V=np.zeros(pasos+1, dtype=float)
U=np.zeros(pasos+1, dtype=float)
I=np.zeros(pasos+1, dtype=float)
# inicializo
#m[1]=0
#h[1]=0
#n[1]=0
V[0]=-60
U[0]=-50

def alfam(v):
    return 0.1*(v+40)/(1-np.exp(-0.1*(v+40)))
def alfah(v):
    return 0.07*np.exp(-0.05*(v+65))
def alfan(v):
    return 0.01*(v+55)/(1-np.exp(-0.1*(v+55)))

def betam(v):
    return 4*np.exp(-0.0556*(v+65))
def betah(v):
    return 1/(1+np.exp(-0.1*(v+35)))
def betan(v):
    return 0.125*np.exp(-0.0125*(v+65))


def minf(v):
    return alfam(v)/(alfam(v)+betam(v))
def hinf(v):
    return alfah(v)/(alfah(v)+betah(v))
def ninf(v):
    return alfan(v)/(alfan(v)+betan(v))

def taum(v):
    return 1/(alfam(v)+betam(v))
def tauh(v):
    return 1/(alfah(v)+betah(v))
def taun(v):
    return 1/(alfan(v)+betan(v))


for i in range(pasos):
    I[i]=0.002*float(i)
    vtaum[i+1]=taum(V[i])
    vtauh[i+1]=tauh(V[i])
    vtaun[i+1]=taun(V[i])

    m[i+1]=minf(V[i])
    h[i+1]=hinf(U[i])
    n[i+1]=ninf(U[i])

    AA=4*36*(n[i+1]**3)*(V[i]+77)*((ninf(V[i])-ninf(U[i]))/vtaun[i+1]) \
        +120*(m[i+1]**3)*(V[i]-50)*((hinf(V[i])-hinf(U[i]))/vtauh[i+1])
        
    BB=4*36*(n[i+1]**3)*(V[i]+77)*(n[i+1]-n[i])/(U[i]-U[i-1]) \
        +120*(m[i+1]**3)*(V[i]-50)*(h[i+1]-h[i])/(U[i]-U[i-1])

    F=0.3*(V[i]+54.4)+36*(n[i+1]**4)*(V[i]+77)+120*(h[i+1])*(m[i+1]**3)*(V[i]-50)
    V[i+1]=V[i]+dt*(-F+I[i])
    U[i+1]=U[i]+dt*(AA/BB)


# ploteamos

plt.plot(tiempo,V)
plt.xlabel("tiempo (ms)")
plt.ylabel("voltaje (mV)")
plt.xlim(0,1000)
plt.show()

plt.plot(tiempo,I)
plt.xlabel("tiempo (ms)")
plt.ylabel("intensidad (nA)")
plt.xlim(0,1000)
plt.show()

plt.plot(V,n)
plt.xlabel("V")
plt.ylabel("n")
plt.xlim(0,1000)
plt.show()

plt.plot(tiempo,m, label="m")
plt.plot(tiempo,h, label="h")
plt.plot(tiempo,n, label="n")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.xlabel("V")
plt.xlim(0,1000)
plt.show()