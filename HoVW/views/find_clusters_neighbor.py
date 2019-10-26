import numpy as np
from scipy.cluster import hierarchy

Z = np.load("Master/data/dendogram.npy")

clusters = []
k = 34

clusters.append(k)
#print(clusters)

asd = True

while(asd == True):
    if(len(clusters)==458):
        break
    x = 1; y = 0
    for z,i in zip(Z, range(Z.shape[0])):
        if(int(z[0]) == k):
            if(z[1] < Z.shape[0]): #caso folha
                clusters.append(int(z[1]))
                #print(clusters)
                k = Z.shape[0] + i+1
            else:
                idx = int((z[1]%Z.shape[0])-1)
                if(Z[idx][0] < Z.shape[0] and Z[idx][1] < Z.shape[0]): # filhos folha-folha
                    clusters.append(int(Z[idx][0]))
                    clusters.append(int(Z[idx][1]))
                    #print(clusters)
                    k = Z.shape[0] + i+1
                elif(Z[idx][0] < Z.shape[0]): # filhos folha-nfolha
                    clusters.append(int(Z[idx][0]))
                    #print(clusters)
                    k = int((Z[idx][1]%Z.shape[0])-1)
                elif(Z[idx][1] < Z.shape[0]): # filhos nfolha-folha
                    clusters.append(int(Z[idx][1]))
                    #print(clusters)
                    k = int((Z[idx][0]%Z.shape[0])-1)
                else:
                    k = int((Z[idx][x]%Z.shape[0])-1)
                    print("saiu 1", x)
                    x = 0

        elif(int(z[1]) == k): 
            if(z[0] < Z.shape[0]): #caso folha
                clusters.append(int(z[0]))
                #print(clusters)
                k = Z.shape[0] + i+1
            else:
                idx = int((z[0]%Z.shape[0])-1)
                if(Z[idx][0] < Z.shape[0] and Z[idx][1] < Z.shape[0]): # caso folha-folha
                    clusters.append(int(Z[idx][0]))
                    clusters.append(int(Z[idx][1]))
                    #print(clusters)
                    k = Z.shape[0] + i+1
                elif(Z[idx][0] < Z.shape[0]): # filhos folha-nfolha
                    clusters.append(int(Z[idx][0]))
                    #print(clusters)
                    k = int((Z[idx][1]%Z.shape[0])-1)
                elif(Z[idx][1] < Z.shape[0]): # filhos nfolha-folha
                    clusters.append(int(Z[idx][1]))
                    #print(clusters)
                    k = int((Z[idx][0]%Z.shape[0])-1)
                else:
                    k = int((Z[idx][y]%Z.shape[0])-1)
                    print("saiu 2", y)
                    y = 1

print(clusters)
        
