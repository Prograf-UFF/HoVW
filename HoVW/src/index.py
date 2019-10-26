import pickle, argparse, re, sys
from os import listdir, path
from utils import Log
from image import Image

def img_structuring(img_path, descriptor_file):
    """Create the descriptor file of image's shapes"""
    try:
        img = Image(img_path)
        shapes = img.tree.get_tree_masks()[1:]
        with open(descriptor_file, 'w') as descriptor_file:
            print("Writing " + descriptor_file.name)
            descriptor_file.write(str(shapes[0].feature_vector.shape[0]) + '\n')
            descriptor_file.write(str(len(shapes)) + '\n')
            for shape in shapes:
                descriptor_file.write(' '.join(str(e) for e in shape.feature_vector.tolist()) + '\n')
    except: raise

    return img, len(shapes)

def init_args():
    """Input Parameters"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='path with images')
    parser.add_argument('-o', help='output path')
    parser.add_argument('-l', help='logs directory')
    parser.add_argument('-d', help='data file path')
    parser.add_argument('-t', help='train or test')

    return parser.parse_args()

def main():
    args = init_args()

    i_path = args.i
    o_path = args.o
    d_path = path.join(o_path, 'descriptor')
    logs_path = args.l
    data_file = args.d
    t_type = args.t

    log = Log(path=logs_path, name='descriptor')

    with open(path.join(data_file, 'img_trees-' + t_type + '.pickle'), 'wb') as handle:
        print("Writing " + handle.name)
        total_shapes = 0
        i = 0
        for img_path in listdir(i_path):
            try:
                img_name = re.sub('\..*', '', img_path)
                img, count_shapes = img_structuring(path.join(i_path, img_path), path.join(d_path, img_name + '.descriptor'))
                total_shapes += count_shapes
                pickle.dump(img, handle, protocol=pickle.HIGHEST_PROTOCOL)
                i += 1
            except KeyboardInterrupt as err:
                log.write(error="Ctrl + c interruption", data=img_path)
                print("Ctrl + c interruption")
                break
            except Exception as e:
                log.write(error=e, data=img_path)
                print("ERROR: " + img_path)
                print(e)
                continue

    print("Images Count: {}; Images Considered {}; Shapes identified: {}".format(len(listdir(i_path)), i, total_shapes))
    log.close()

if __name__ == "__main__":
    main()