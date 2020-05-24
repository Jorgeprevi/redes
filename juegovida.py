import numpy as np
import matplotlib.pyplot as plt
import random
import time

# tamaño de red
L=7
#numero de pasos
N=100
#densidad inicial de ocupados
inidens=0.3

def pbc(a):
    if a<0:
        return L-1
    elif a==L:
        return 0
    else:
        return a

def cercanos(m, x, y):
    '''
    Devuelve la cantidad de vecinos cercanos con pbc
    '''
    n=0
    for i in range(-1,2):   
        for j in range(-1,2):
            if (i,j)!=(0,0):
                n=n+m[pbc(x+i),pbc(y+j)]
    return n

m=np.zeros([L,L], dtype=int)
mpre=np.zeros([L,L], dtype=int)

# inicializacion aleatoria
#for i in range(L):
#    for j in range(L):
#        if random.uniform(0,1)<inidens:
#            m[i,j]=1
#        else:
#            m[i,j]=0

#cruz
# m[11,13]=1
# m[12,13]=1
# m[13,13]=1

#osc2
# m[2,5]=1
# m[2,6]=1
# m[3,5]=1
# m[4,8]=1
# m[5,8]=1
# m[5,7]=1

#glider
m[2,3]=1
m[3,4]=1
m[4,4]=1
m[4,3]=1
m[4,2]=1



print(m)
mpre=np.copy(m)

# realizo N pasos
for i in range(N):
    for j in range(L):
        for k in range(L):
            nn=cercanos(mpre,j,k)
            if nn<2 and mpre[j,k]==1:
                m[j,k]=0
            elif nn>=4 and mpre[j,k]==1:
                m[j,k]=0
            elif nn==3 and mpre[j,k]==0:
                m[j,k]=1
    mpre=np.copy(m)
    if i>0:
        fig=plt.matshow(m)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.show()
        time.sleep(1)
    











        
    
            