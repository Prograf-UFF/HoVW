import os, argparse, random

parser = argparse.ArgumentParser()
parser.add_argument('-i', help='path with images')
parser.add_argument('-o', help='path with images')
args = parser.parse_args()

i_path = args.i
o_path = args.o

images = os.listdir(i_path)

test_size = int(len(images) * 0.3)
test_index = random.sample(range(len(images)), k=test_size)

testgroup = []
removed = 0
for i in test_index:
    testgroup.append(images.pop(i-removed))
    removed += 1

# val_size = int(len(images) * 0.2) + 1
# val_index = random.sample(range(len(images)), k=val_size)

# valgroup = []
# removed = 0
# for i in test_index:
#     valgroup.append(images.pop(i-removed))
#     removed += 1

traingroup = images

try:
    train_folder = o_path + "/train"
    print("Creating " + train_folder)
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    test_folder = o_path + "/test"
    print("Creating " + test_folder)
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)
    # test_folder = o_path + "/validation"
    # print("Creating " + test_folder)
    # if not os.path.exists(test_folder):
    #     os.makedirs(test_folder)
except Exception as e:
    print("Error:", e)

print("Creating Train Set")
for i in traingroup:
    try:
        print("Moving ", i)
        os.rename(i_path + "/" + i, train_folder + "/" + i)
    except Exception as e:
        print("Error:", e)

print("Creating Test Set")
for i in testgroup:
    try:
        print("Moving ", i)
        os.rename(i_path + "/" + i, test_folder + "/" + i)
    except Exception as e:
        print("Error:", e)

# print("Creating Validation Set")
# for i in valgroup:
#     try:
#         print("Moving ", i)
#         os.rename(i_path + "/" + i, test_folder + "/" + i)
#     except Exception as e:
#         print("Error:", e)


print("Train size ", len(traingroup))
print("Test size ", len(testgroup))
# print("Validation size ", len(valgroup))
