import os
import numpy as np

k_list = np.arange(100, 1300, 100)
k_list = [100]
for k in k_list:
    with open('exec/results-batch.txt', 'a') as f:
        f.write("-- K = {}\n".format(k))
    with open('exec/results-batch2.txt', 'a') as f:
        f.write("-- K = {}\n".format(k))
    
    q1 = """python src/kmeans-nodes.py -i "Master/datasetOutput/descriptor/" -o "Master/datasetOutput" -c {} -l "Master/datasetOutput/logs" """.format(k)
    print(q1);os.system(q1)
    
    q2 = """python src/kmeans-nodes-test.py -i "Master/datasetOutput/test/descriptor/" -o "Master/datasetOutput/test" -c "Master/data/kme-clusters2.clr" -l "Master/datasetOutput/logs" """
    print(q2);os.system(q2)
    
    q3 = """python src/assign-node-label.py -i "Master/data/img_trees-train.pickle" -c "Master/datasetOutput/BOW/assignment-kme" -d "Master/data" -t "train" """
    print(q3);os.system(q3)
    
    q4 = """python src/assign-node-label.py -i "Master/data/img_trees-test.pickle" -c "Master/datasetOutput/test/BOW/assignment-kme" -d "Master/data" -t "test" """
    print(q4);os.system(q4)
    
    q5 = """python src/trees-distance.py -i "Master/data/img_trees_labeled-train.pickle" -c "Master/data/kme-clusters-cdm.npy" -m "Master/data/tree_distances_matrix-train.npy" -d "Master/data/apted_custom_dist.pickle" -p 4 """
    print(q5);os.system(q5)
    
    #bw_list = [None, 0.3, 0.1, 0.01]
    bw_list = [0.01, 0.1, 0.3, 0.5, None,0.7]
    bw_list = [0.7]
    for bw in bw_list:
        with open('exec/results-batch.txt', 'a') as f:
            f.write("-- Bandwidth = {}\n".format(bw))
        with open('exec/results-batch2.txt', 'a') as f:
            f.write("-- Bandwidth = {}\n".format(bw))
        
        #ATENCAO: ADICIONEI NO CODIGO PARA ESCREVER O BANDWIDTH no arquivo de batch
        q6 = """python src/meanshift-graphs.py "Master/data/tree_distances_matrix-train.npy" "Master/data/trees_clustered" {} "Master/data/img_trees_labeled-train.pickle" """.format(bw)
        print(q6);os.system(q6)
        
        q7 = """python src/assign-tree-label.py "Master/data/img_trees_labeled-train.pickle" "Master/data/trees_clustered_vq-train.npy" """
        print(q7);os.system(q7)
        
        q8 = """python src/meanshift-graphs-test.py "Master/data/img_trees_labeled-test.pickle" "Master/data/trees_clustered_codebook.npz" "Master/data/kme-clusters-cdm.npy" "Master/data/trees_clustered_centers.npy" """
        print(q8);os.system(q8)
        
        q9 = """python src/centers_dendrogram.py -d "Master/data/tree_distances_matrix-train.npy" -c "Master/data/trees_clustered_centers.npy" -o "Master/data/" """
        print(q9);os.system(q9)
        
        q10 = """python views/reveal_clusters-graphs.py"""
        print(q10);os.system(q10)
        
        # # p_list = [20, 15, 10, 5, 1]
        # p_list = [20]
        # for p in p_list:
        #     q11 = """python src/precison_recall_test2.py {}""".format(p)
        #     print(q11);os.system(q11)

