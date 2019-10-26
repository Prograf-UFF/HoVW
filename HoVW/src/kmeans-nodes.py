#!/usr/bin/env python

import pickle, argparse, re, time
from os import listdir, path
import numpy as np
from utils import Log
from clusters import Cluster

from sklearn import metrics
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt

def init_args():
    """Input Parameters"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help="path with image's descriptors")
    parser.add_argument('-o', help='output path')
    parser.add_argument('-c', help='codebook size')
    parser.add_argument('-l', help='logs directory')

    return parser.parse_args()

def read_image_descriptors(i_path, log):
    """Retrive all features from all image's descriptor file and build
    the train set X"""
    X = np.asarray([[None]*29], dtype=np.float_) #HARD-CODED 29 Features
    y = []
    for dfile in listdir(i_path):
        try:
            with open(path.join(i_path, dfile), 'r') as f:
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
            log.write(error=e, data=dfile)
            print("ERROR: " + dfile)
            print(e)
            break
            continue
    
    print(X.shape)
    return X, y

def main():
    args = init_args()

    i_path = args.i
    o_path = args.o
    logs_path = args.l
    k_clusters = int(args.c)
    
    log = Log(logs_path, 'clustering-kme')

    X, y = read_image_descriptors(i_path, log)

    X = np.delete(X, 0, 0)

    # TODO: SEPARAR AVALIAÇÃO EM UM OUTRO ARQUIVO. ATUALMENTE SE QUISER
    # AVALIAÇÃO OU EXECUÇÃO NECESSITA APENAS COMENTAR/DESCOMENTAR A SEÇÃO
    
    #AVALIAÇÃO
    # #--- supervisionada
    # sls = []
    # chs = []

    #--- não supervisionada
    # inertia_list = []

    # kc = [1000, 1500, 2000, 2500, 3000]
    # for k_clusters in kc:
    #     print(k_clusters)
    #     kmeans = Cluster(method='KMeans', init='k-means++', n_clusters=k_clusters, random_state=42, max_iter=300)
    #     kmeans.fit(X)
        
        # #--- supervisionada
        # labels = kmeans.labels_
        # sls.append(metrics.silhouette_score(X, labels, metric='mahalanobis'))
        # chs.append(metrics.calinski_harabaz_score(X, labels))

        #--- não supervisionada
        # inertia_list.append(kmeans.inertia_)
    
    # fig = plt.figure()

    # #--- supervisionada
    # plt.subplot(1,1,1)
    # plt.title('silhouette_score')
    # plt.xlabel('codebook size')
    # plt.plot(kc, sls, 'r')

    # plt.subplot(2,1,2)
    # plt.title('calinski_harabaz_score')
    # plt.xlabel('codebook size')
    # plt.plot(kc, chs, 'b')

    # print(sls)
    # print(chs)

    #--- não supervisionada supervisionada
    # plt.subplot(1,1,1)
    # plt.title('inertia')
    # plt.xlabel('codebook size')
    # plt.plot(kc, inertia_list, 'r')

    # print(inertia_list)

    # plt.savefig('kmeans-scores.png')
    # plt.show()

    # EXECUÇÃO
    kmeans = Cluster(method='KMeans', init='k-means++', n_clusters=k_clusters, batch_size=X.shape[0], random_state=42, max_iter=300)
    print("Learning...")
    kmeans.fit(X)
    
    kmeans.save('Master/data/', 'kme-clusters')

    # with open(path.join(o_path, 'codebook-bovw-kme.cb'), 'w') as f:
    #     print("Writing " + f.name)
    #     f.write(str(kmeans.cluster_centers_.shape[1]) + '\n' +
    #     str(kmeans.cluster_centers_.shape[0]) + '\n')
    #     for c_center in kmeans.cluster_centers_:
    #         f.write(' '.join(str(e) for e in c_center.tolist()) + '\n')

    log.close()

    ###Assignment --- Vector quantization
    log = Log(logs_path, 'bag-assignment-kme')

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

    # ### TODO:data for classifier -- em um ambiente não supervisionado deve estar comentado
    im_names = {}
    im_features = np.zeros((len(oufl), k_clusters), np.float_)
    name_c = 0
    # ###

    quantization = kmeans.predict(X)
    j = 0

    t0 = time.time()
    for e in oufl:
        bag = [0] * k_clusters
        with open(path.join(o_path, 'BOW/assignment-kme/' + e[0] + '.codeword'), 'w') as fa, \
            open(path.join(o_path, 'BOW/bag-kme/' + e[0] + '.bagw'), 'w') as fb:
            print("Writing " + fa.name + " and " + fb.name)
            for i in range(e[1]):
                vq = quantization[j]
                j += 1
                bag[int(vq)] += 1
                # ### ### TODO:data for classifier
                im_names[name_c] = e[0]
                im_features[name_c][int(vq)] += 1
                # ###
                fa_string = str(vq) + ' '
                fa_string += ' '.join(str(k) for k in kmeans.cluster_centers_[vq]) + '\n'
                fa.write(fa_string)               
            # ### ### TODO:data for classifier
            name_c += 1
            # ###
            sum_bag = sum(bag)
            fb.write(' '.join(str(k/sum_bag) for k in bag))

    # ### ### TODO:data for classifier
    with open('Master/data/classf_data-train.pickle', 'wb') as handle:
        pickle.dump(im_names, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(im_features, handle, protocol=pickle.HIGHEST_PROTOCOL)
    ###
    

    log.close()

if __name__ == "__main__":
    main()