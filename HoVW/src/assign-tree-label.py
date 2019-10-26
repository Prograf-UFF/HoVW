import sys, os, pickle
import numpy as np
import image as Image

trees_file = sys.argv[1]
vq_file = sys.argv[2]

vq_list = np.load(vq_file)

img_trees = []
test = []
with open(trees_file, 'rb') as f:
    while True:
        try:
            img_trees.append(pickle.load(f))
        except EOFError:
            break

for img, vq in zip(img_trees, vq_list):
    print(img.tree.name, int(vq))
    test.append((img.tree.name, int(vq)))
    img.tree.label = int(vq)
'''
print("----")
for a in test:
    if (a[1] == 85):
        print(a[0])
'''
with open('Master/data/img_trees_labeled_full-train.pickle', 'wb') as handle:
    for img_tree in img_trees:
        pickle.dump(img_tree, handle, protocol=pickle.HIGHEST_PROTOCOL)
