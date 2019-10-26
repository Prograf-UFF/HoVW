import argparse, pickle, re
from sklearn.svm import LinearSVC

parser = argparse.ArgumentParser()
parser.add_argument('-i', help='path with data train')
parser.add_argument('-o', help='path with data test')
args = parser.parse_args()

i_path = args.i
o_path = args.o

objects = []
with (open(i_path, "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break

im_features = objects[1]

im_names = []
for name in objects[0]:
    im_names.append(re.sub('-.*', '', objects[0][name]))

print(im_features.shape, len(im_names))

clf = LinearSVC()
clf.fit(im_features, im_names)

objects = []
with (open(o_path, "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break

im_features = objects[1]

im_names = []
for name in objects[0]:
    im_names.append(re.sub('-.*', '', objects[0][name]))

print(im_features.shape, len(im_names))

print(clf.score(im_features, im_names))
