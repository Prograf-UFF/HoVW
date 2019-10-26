from sklearn.cluster import KMeans
import numpy as np
from os import listdir, path
import argparse, time, re, pickle

parser = argparse.ArgumentParser()
parser.add_argument('-i', help='path with images')
parser.add_argument('-o', help='output path')
parser.add_argument('-c', help='clusters')
parser.add_argument('-l', help='logs directory')

args = parser.parse_args()

i_path = args.i
o_path = args.o
logs_path = args.l
k_clusters = -1
c_path = args.c
X = np.asarray([[None]*29], dtype=np.float_) #HARD-CODED TEM QUE LER ESSE 29 DO ARQUIVO
y = []

for dfile in listdir(i_path):
    try:
        with open(i_path+dfile, 'r') as f:
            print("Reading from " + f.name)
            content = f.readlines()

        for i in range(int(content[1])):
            X = np.append(X,
                [np.array(list(np.float_(e) for e in content[2+i].split(' ')), dtype=np.float_)],
                axis=0)
            y.append(re.sub('\..*', '', dfile))
    
    except KeyboardInterrupt as err:
            print("Ctrl + c interruption")
            break
    except Exception as e:
        print("ERROR: " + dfile)
        print(e)
        break
        continue

X = np.delete(X, 0, 0)

objects=[]
with (open(c_path, "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break

###Assignment --- Vector quantization

#PRECISA SER TESTADO EM MAIS CASOS-------
oufl = []
past = None
count = 0
for i in range(len(y)):
    if(y[i] != past):
        oufl.append((past, count))
        count = 1
        past = y[i]
    else: count+=1
    past = y[i]
oufl.append((past, count))
oufl = oufl[1:]
#-----
cluster = objects[0]
k_clusters = cluster.get_params()['n_clusters']
quantization = cluster.predict(X)

j = 0
im_names = {}
im_features = np.zeros((len(oufl), k_clusters), np.float_)
name_c = 0

for e in oufl:
    bag = [0] * k_clusters
    with open(path.join(o_path, 'BOW/assignment-kme/' + e[0] + '.codeword'), 'w') as fa, \
        open(path.join(o_path, 'BOW/bag-kme/' + e[0] + '.bagw'), 'w') as fb:
        #print("Writing " + fa.name + " and " + fb.name)
        for i in range(e[1]):
            vq = quantization[j] 
            bag[int(vq)] += 1
            ###
            im_names[name_c] = e[0]
            im_features[name_c][int(vq)] += 1
            ###
            fa.write(str(vq) + ' ')
            fa.write(' '.join(str(k) for k in cluster.cluster_centers_[vq].tolist()) + '\n')
            j += 1
        ###
        name_c += 1
        ###
        bagf = [k/sum(bag) for k in bag]
        fb.write(' '.join(str(k) for k in bagf))

with open('Master/data/classf_data-test.pickle', 'wb') as handle:
    pickle.dump(im_names, handle, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(im_features, handle, protocol=pickle.HIGHEST_PROTOCOL)