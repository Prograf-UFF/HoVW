:: sh build_master_dir.sh

SET K=250
SET BANDWIDTH=0.3
SET PROCESS=2

:: python classification/split-train-test.py -i "Master/images" -o "Master/dataset"

python src/index.py -i "Master/dataset/train/" -o "Master/datasetOutput" -l "Master/datasetOutput/logs" -d "Master/data" -t "train"
:: python src/index.py -i "Master/dataset/test/" -o "Master/datasetOutput/test" -l "Master/datasetOutput/logs" -d "Master/data" -t "test"

python src/kmeans-nodes.py -i "Master/datasetOutput/descriptor/" -o "Master/datasetOutput" -c %K% -l "Master/datasetOutput/logs"
:: python classification/kmeans-nodes-test.py -i "Master/datasetOutput/test/descriptor/" -o "Master/datasetOutput/test" -c "Master/data/kme-clusters.clr" -l "Master/datasetOutput/logs"

:: python src/others/meanshift-nodes.py -i "Master/datasetOutput/descriptor/" -o "Master/datasetOutput" -b 0.3 -l "Master/datasetOutput/logs"
:: python classification/meanshift-test.py -i "Master/datasetOutput/test/descriptor/" -o "Master/datasetOutput/test" -c "msh-clusters.clr" -l "Master/datasetOutput/logs"

python src/assign-node-label.py -i "Master/data/img_trees-train.pickle" -c "Master/datasetOutput/BOW/assignment-kme" -d "Master/data" -t "train"
:: python src/assign-node-label.py -i "Master/data/img_trees-test.pickle" -c "Master/datasetOutput/test/BOW/assignment-kme" -d "Master/data" -t "test"

python src/trees-distance.py -i "Master/data/img_trees_labeled-train.pickle" -c "Master/data/kme-clusters-cdm.npy" -m "Master/data/tree_distances_matrix-train.npy" -d "Master/data/apted_custom_dist.pickle" -p %PROCESS%

python src/meanshift-graphs.py "Master/data/tree_distances_matrix-train.npy" "Master/data/trees_clustered" %BANDWIDTH% "Master/data/img_trees_labeled-train.pickle"

python src/assign-tree-label.py "Master/data/img_trees_labeled-train.pickle" "Master/data/trees_clustered_vq-train.npy"
:: python src/others/meanshift-graphs-test.py "Master/data/img_trees_labeled-test.pickle" "Master/data/trees_clustered_codebook.npz" "Master/data/kme-clusters-cdm.npy" "Master/data/trees_clustered_centers.npy"

python src/centers_dendrogram.py -d "Master/data/tree_distances_matrix-train.npy" -c "Master/data/trees_clustered_centers.npy" -o "Master/data/"

::EXTRAS -- visualization only, ignore on the main pipeline

:: python databases/populate.py -f "Master/data/img_trees_labeled_full-train.pickle"
:: python classification/classify-nodes.py -i "Master/data/classf_data-train.pickle" -o "Master/data/classf_data-test.pickle"
python views/reveal_clusters-graphs.py
python views/reveal_clusters-nodes.py
:: python views/visualize_graph_spatial_distribution.py

::FIT
:: python src/fit.py -i "Master/images/olho.png" -o "Master/images/tmp/clusters.potato" -l "Master/datasetOutput/logs" -a "Master/data/kme-clusters.clr" -t "Master/data/trees_clustered_codebook.npz" -d "Master/data/apted_custom_dist.pickle" -z "Master/data/dendogram.npy"