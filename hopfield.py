import numpy as np
import matplotlib.pyplot as plt
import random
import sys
np.set_printoptions(threshold=sys.maxsize)
E=150
pasos=1500
L=50
T=np.zeros(E)
patron=np.zeros(L)
# matriz de pesos
w=np.zeros([L,L])
# vector de estados
s=np.zeros(L)
spre=np.zeros(L)
# potenciales de membrana
v=np.zeros(L)
# vector de solapamiento
solapamiento=np.zeros(E)

xaxis=np.linspace(0,E,E)

def pot(a, s, m):
    '''
    devuelve el potencial de membrana de neurona a. s=vector estado, m=matriz pesos
    '''
    suma=0.0
    for i in range(L):
        if i!=a:
            suma+=s[i]*m[i,a]
    return suma

def sigmoide(x, umbral, temperatura):
    '''
    devuelve el valor de la sigmoide centrada en umbral para un potencial x y una T
    '''
    c=1/(1+np.exp(-(x-umbral)/temperatura))
    return c

def solapa(s1, s2, L):
    '''
    calcula el solapamiento entre los vectores estado s1 y s2
    '''
    sol=0.0
    for i in range(L):
        if s2[i]==s1[i]:
            sol=sol+1
        else:
            sol=sol-1
    sol/=L
    return sol

# inicializamos el patron y la matriz de pesos
for i in range(L):
    patron[i]=i%2

for i in range(L):
    for j in range(L):
        w[i,j]=(2*patron[i]-1)*(2*patron[j]-1)
        if i==j:
            w[i,j]=0

i_lower=np.tril_indices(L-1, -1)
w[i_lower]=w.T[i_lower]


T[0]=15
for k in range(E):
    for i in range(L):
        if random.uniform(0,1)<0.5:
            s[i]=1
        else:
            s[i]=0
    spre=np.copy(s)

    for i in range(pasos):
        for j in range(L):
            a=random.uniform(0,1)
            if a<sigmoide(pot(j,spre,w),0,T[k]):
                s[j]=1
            else:
                s[j]=0
        solapamiento[k]+=solapa(patron, spre, L)
        spre=np.copy(s)
    solapamiento[k]/=pasos
    if k!=E-1:
        T[k+1]=T[k]-0.05

#print (w)
plt.scatter(T,solapamiento, s=2)
plt.xlabel("T")
plt.ylabel("solapamiento")
plt.grid()
plt.show()