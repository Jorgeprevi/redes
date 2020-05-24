import numpy as np
import matplotlib.pyplot as plt
import random

L=2000     # tamaño de red
pasos=1
tiempo=np.linspace(0, pasos-1, pasos)
def pbc(x, N):
    if x>N:
        return x-N-1 
    elif x<0:
        return N+x+1
    else:
        return x

# inicializamos red de barabasi albert mediante preferential attachment
MB=np.zeros([L,L], dtype=int)    # matriz de adyacencia red BA
# inicializamos los 5 primeros nodos con 2 uniones por nodo
for i in range(5):
    MB[i,pbc(i+2,4)]+=1
    MB[pbc(i+2,4),i]+=1
# llevamos a cabo el proceso de pref att con una arista por nuevo nodo
for i in range(5,L):
    a=random.randint(0, np.count_nonzero(MB))
    count=0
    for j in range(i):
        for k in range(i):
            if MB[j,k]==1:
                if count==a:
                    MB[j,i]=1
                    MB[i,j]=1
                count=count+1

m=2     # 2m= numero de NN
p=0.05   # prob recableado
MW=np.zeros([L,L], dtype=int)    # matriz de adyacencia red WS
MWpre=np.zeros([L,L], dtype=int)
# inicializamos la red WS
for i in range(L):
    for j in range(-m,m+1):
        MW[pbc(i+j,L-1),i]=1
        MW[i,i]=0
MWpre=np.copy(MW)
elem=np.count_nonzero(MW)
total=0
ind=np.zeros(2*m+1, dtype=int)
for i in range(L):
    count=0
    for k in range(i-m, i+m+1):
        ind[count]=pbc(k, L-1)
        count=count+1
    for j in ind:
        if MWpre[i,j]!=0 and random.uniform(0,1)<p:
            total+=1
            MW[i,j]=0
            MW[j,i]=0
            a=random.randint(m, L-m-1)
            MW[i,pbc(i+a,L-1)]=1
            MW[pbc(i+a,L-1),i]=1

# print(np.count_nonzero(MW))

# inicio el proceso

SB=np.zeros(L)      # vector estado ba
SW=np.zeros(L)      # vector estado ws
NNB=np.zeros(L)     # cantidad de vecinos cercanos de cada nodo en ba
NNW=np.zeros(L)     # igual en ws
magB=np.zeros(pasos)
magW=np.zeros(pasos)
roB=np.zeros(pasos)   # densidad de interfases red BA
roW=np.zeros(pasos)   # igual en WS
pkB=np.zeros(L)
pkW=np.zeros(L)
axis=np.zeros(L)
for i in range(L):
    axis[i]=i

for i in range(L):
    NNB[i]=MB.sum(axis=0)[i]
    NNW[i]=MW.sum(axis=0)[i]
    if random.uniform(0,1)<0.5:
        SB[i]=1
        SW[i]=1
    else:
        SB[i]=-1
        SW[i]=-1

for i in range(L):
    pkB[int(NNB[i])]+=1
    pkW[int(NNW[i])]+=1
pkB/=sum(pkB)
pkW/=sum(pkW)

plt.bar(axis, pkB, label="Barabasi")
plt.legend(bbox_to_anchor=(0.7, 0.95), loc='upper left', borderaxespad=0.)
plt.xlim(0,80)
plt.yscale('log')
plt.ylabel("p(k)")
plt.xlabel("k")
plt.show()

plt.bar(axis, pkW, label="Watts")
plt.legend(bbox_to_anchor=(0.7, 0.95), loc='upper left', borderaxespad=0.)
plt.xlim(0,80)
plt.ylabel("p(k)")
plt.xlabel("k")
plt.show()

# proceso BA
for k in range(pasos):
    magB[k]=sum(SB)
    np=float(sum(sum(MB))/2)
    na=0.0
    for l in range(L):
        for m in range(L):
            if MB[l,m]==1 and SB[l]!=SB[m]:
                na=na+1.0
    roB[k]=na/(2*np)
    i=random.randint(0, L-1)
    a=random.randint(0, NNB[i])
    count=0
    for j in range(L):
        if MB[i,j]==1:
            if count==a:
                SB[i]=SB[j]
            count+=1

# proceso WS
for k in range(pasos):
    magW[k]=sum(SW)
    np=float(sum(sum(MW))/2)
    na=0.0
    for l in range(L):
        for m in range(L):
            if MW[l,m]==1 and SW[l]!=SW[m]:
                na=na+1.0
    roW[k]=na/(2*np)
    i=random.randint(0, L-1)
    a=random.randint(0, NNW[i])
    count=0
    for j in range(L):
        if MW[i,j]==1:
            if count==a:
                SW[i]=SW[j]
                #print(SW)
            count+=1      

plt.plot(tiempo, magB, label="B")
plt.plot(tiempo, magW, label="W")
plt.legend(bbox_to_anchor=(0.85, 0.85), loc='upper left', borderaxespad=0.)
plt.ylabel("magnetización")
plt.xlabel("paso")
plt.show()

plt.plot(tiempo, roB, label="B")
plt.plot(tiempo, roW, label="W")
plt.legend(bbox_to_anchor=(0.85, 0.85), loc='upper left', borderaxespad=0.)
plt.yscale('log')
plt.ylabel("n_A")
plt.xlabel("paso")
plt.show()