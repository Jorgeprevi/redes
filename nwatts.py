import numpy as np
import matplotlib.pyplot as plt
import random
import sys 

N=100
m=2
pasos=40
p=np.zeros(pasos)
axis=np.zeros(N)
for i in range(N):
    axis[i]=i


Vre=np.zeros(N, dtype=int)    # vector de nodos
Vat=np.zeros(N, dtype=int)
A=np.zeros([N,N], dtype=int)  # matriz de adyacencia inicial
RECAB=np.zeros([N,N], dtype=int) # matriz adya recableada
RECABaux=np.zeros([N,N], dtype=int)
RECAB2=np.zeros([N,N], dtype=int)
ATAJ=np.zeros([N,N], dtype=int) # matriz adya con atajos
ATAJaux=np.zeros([N,N], dtype=int)
ATAJ2=np.zeros([N,N], dtype=int)
pkR=np.zeros(N) # array de distr de prob de grad
pkA=np.zeros(N)
la=np.zeros(pasos)
lr=np.zeros(pasos)
Ca=np.zeros(pasos)
Cr=np.zeros(pasos)

def pbc(x, N):
    if x>N:
        return x-N-1 
    elif x<0:
        return N+x+1
    else:
        return x

def dijkstra(A, Vector): 
  
    dist=10000*Vector 
    sptSet=False*Vector
  
    for cosa in range(len(Vector)): 
        min = 10000
        for vh in range(len(Vector)): 
            if dist[vh]<min and sptSet[vh]==False: 
                min=dist[vh] 
                min_index=vh
        u=min_index
        sptSet[u]=True
        for vj in range(len(Vector)): 
            if A[u,vj]>0 and sptSet[vj]==False and dist[vj]>dist[u]+A[u,vj]: 
                dist[vj]=dist[u]+A[u,vj] 
    return dist

# defino el vector de vertices
for i in range(N):
    Vre[i]=i
    Vat[i]=i
for i in range(pasos):
        p[i]=0.02*i

for pas in range(pasos):

    # inicializo matriz inicial
    for i in range(N):
        for j in range(-m,m+1):
            A[pbc(i+j,N-1),i]=1
            A[i,i]=0

    # genero las matrices con recableado y con atajos
    # el recableado se hace con vecinos no cercanos

    # recableado
    ind=np.zeros(2*m+1, dtype=int)
    RECAB=np.copy(A)
    for i in range(N):
        count=0
        for k in range(i-m, i+m+1):
            ind[count]=pbc(k, N-1)
            count=count+1
        for j in ind:
            if A[i,j]!=0 and random.uniform(0,1)<p[pas]:
                RECAB[i,j]=0
                RECAB[j,i]=0
                a=random.randint(m, N-m-1)
                RECAB[i,pbc(i+a,N-1)]=1
                RECAB[pbc(i+a,N-1),i]=1

    # atajos
    ATAJ=np.copy(A)
    for i in range(N):
        count=0
        for k in range(i-m, i+m+1):
            ind[count]=pbc(k, N-1)
            count=count+1
        for j in ind:
            if A[i,j]!=0 and random.uniform(0,1)<p[pas]:

                a=random.randint(m, N-m-1)
                ATAJ[i,pbc(i+a,N-1)]=1
                ATAJ[pbc(i+a,N-1),i]=1

    # calculo las distancias medias por el algoritmo de dijkstra
    for i in range(N):
        aR=dijkstra(RECAB, Vre)
        aA=dijkstra(ATAJ, Vat)
        lr[pas]=lr[pas]+float(sum(aR))/float(N-1)
        la[pas]=la[pas]+float(sum(aA))/float(N-1)
        # desplazo los índices de las matriz de adyacencia en cada iteración
        for j in range(N):
            for k in range(N):
                RECABaux[j,k]=RECAB[pbc(j-1,N-1),pbc(k-1,N-1)]
                #print(j,k,pbc(j-1,N-1), pbc(k-1,N-1))
                ATAJaux[j,k]=ATAJ[pbc(j-1,N-1),pbc(k-1,N-1)]

        RECAB=np.copy(RECABaux)
        ATAJ=np.copy(ATAJaux)
    lr[pas]=lr[pas]/float(N)
    la[pas]=la[pas]/float(N)

    # calculo los coef de clustering
    denomr=0.0
    denoma=0.0
    RECAB2=np.linalg.matrix_power(RECAB,2)
    ATAJ2=np.linalg.matrix_power(ATAJ,2)
    for j in range(N):
        for k in range(N):
            if j!=k:
                denomr+=RECAB2[k,j]
                denoma+=ATAJ2[k,j]
    Cr[pas]=(np.trace(np.linalg.matrix_power(RECAB,3))/denomr)
    Ca[pas]=(np.trace(np.linalg.matrix_power(ATAJ,3))/denoma)

    # calculo las distribucion de prob de grados
    distR=np.zeros(N, dtype=int)
    distA=np.zeros(N, dtype=int)
    for i in range(N):
        for j in range(N):
            distR[i]+=RECAB[i,j]
            distA[i]+=ATAJ[i,j]
    for i in range(N):
        pkR[distR[i]]+=1
        pkA[distA[i]]+=1
    pkR/=sum(pkR)
    pkA/=sum(pkA)

    if pas==pasos-1:
        plt.bar(axis, pkR, label="Re")
        plt.legend(bbox_to_anchor=(0.85, 0.85), loc='upper left', borderaxespad=0.)
        plt.ylabel("p(k)")
        plt.xlabel("k")
        plt.show()

        plt.bar(axis, pkA, label="At")
        plt.legend(bbox_to_anchor=(0.85, 0.85), loc='upper left', borderaxespad=0.)
        plt.ylabel("p(k)")
        plt.xlabel("k")
        plt.show()

plt.plot(p, la, label="At")
plt.plot(p, lr, label="Re")
plt.legend(bbox_to_anchor=(0.85, 0.85), loc='upper left', borderaxespad=0.)
plt.ylabel("l")
plt.xlabel("p")
plt.show()

plt.plot(p, Ca, label="At")
plt.plot(p, Cr, label="Re")
plt.legend(bbox_to_anchor=(0.85, 0.85), loc='upper left', borderaxespad=0.)
plt.ylabel("C")
plt.xlabel("p")
plt.show()