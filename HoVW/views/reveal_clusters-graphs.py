import os, shutil, pickle, sys, re, collections
sys.path.insert(0, 'src')
from image import Image

def handle_tree(tree, path):
    name = tree.name
    label = str(tree.label)

    print(name, label)

    dst = os.path.join(path, label)
    if label not in os.listdir(path):
        os.mkdir(dst)
    
    dst += '/' + re.sub('.*\/', '', name)
    
    shutil.copyfile(name, dst)

def retrive_imgs(font):
    img_trees = []
    for t in font:
        f_name = 'Master/data/img_trees_labeled_full-' + t + '.pickle'
        print("Reading from:", f_name)
        with open(f_name, 'rb') as f:
            while True:
                try:
                    img_trees.append(pickle.load(f))
                except EOFError:
                    break
    
    return img_trees

def clusters_len(img_trees):
    d = {}
    for it in img_trees:
        l = it.tree.label
        if l not in d: d[l] = 1
        else: d[l] += 1

    return collections.OrderedDict(sorted(d.items(), key=lambda t: t[0]))

def make_dir(root, relative):

    if 'clusters' not in os.listdir(root):
        os.mkdir(os.path.join(root, 'clusters'))

    clusters = os.path.join(root, 'clusters')

    if relative in os.listdir(clusters):
        shutil.rmtree(os.path.join(clusters, relative))
    
    root_clusters = os.path.join(clusters, relative)
    os.mkdir(root_clusters)
    os.mkdir(os.path.join(root_clusters, "full"))
    os.mkdir(os.path.join(root_clusters, "train"))
    os.mkdir(os.path.join(root_clusters, "test"))

    return root_clusters

def log(font, d):
    f = open(font + '.txt', 'w')
    for k,v in d.items():
        f.write(str(k) + ":" + str(v) + '\n')

def main(): 
    
    root = 'Master'
    relative = 'graphs'
    root_clusters = make_dir(root, relative)

    for t in [['train']]:#, ['test'], ['train', 'test']]:
        img_trees = retrive_imgs(t)

        dst = os.path.join(root_clusters, t[0]) \
        if len(t) == 1 else os.path.join(root_clusters, 'full')

        for it in img_trees:
            handle_tree(it.tree, dst)

        d = clusters_len(img_trees)
        print(d)

        log(os.path.join(dst, 'log'), d)

if __name__ == '__main__':
	main()