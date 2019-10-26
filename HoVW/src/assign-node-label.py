import pickle, argparse, collections, re
from os import listdir, path
import matplotlib.pyplot as plt

def init_args():
    """Input Parameters"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='images idx file')
    parser.add_argument('-c', help='codewords dir')
    parser.add_argument('-d', help='data file path')
    parser.add_argument('-t', help='train or test')

    return parser.parse_args()

def distribuition_plot(agg_dict, dfile_path):
    plt.bar(range(len(agg_dict)), agg_dict.keys(), align="center")
    xitcks = []
    for i in agg_dict.values():
        xitcks.append(len(i))
    plt.xticks(range(len(agg_dict)), xitcks, rotation='vertical')

    plt.xlabel("Count of clusters")
    plt.ylabel("Elements in cluster")

    plt.rcParams["figure.figsize"] = [600,600]
    for a,b in zip(range(len(agg_dict)), agg_dict.keys()):
        plt.text(a-0.5, b, b)
    plt.savefig(path.join(dfile_path, 'nodes-distribuition-plot.png'))
    # plt.show()

def main():
    args = init_args()
    i_file = args.i
    cw_dir = args.c
    t_type = args.t
    dfile_path = args.d
    hist_l = {}
    images = []

    # TODO: Dá para otimizar o uso da memória. Basta, ao invés, de armazenar
    # tudo em images [] ler do arquivo e processar ou upar para memória em batch
    with open(i_file, 'rb') as f:
        while True:
            try:
                images.append(pickle.load(f))
            except EOFError:
                break

    for im in images:
        cw = re.sub('.*\/', '', im.tree.name)
        cw = re.sub('\..*', '', cw) + '.codeword'
        print("Working on", cw, "and", im.tree.name)
        labels = []
        with open(path.join(cw_dir,cw), 'r') as f:
            for line in f:
                label = int(line.split(" ")[0])
                if(label not in hist_l): hist_l[label] = 1
                else: hist_l[label] += 1
                labels.append(label)

        # TODO: juntar as duas funções abaixo; elas deveriam ser feitas juntas
        im.tree.set_labels(labels)
        im.tree.set_nodes_depth()

    with open(path.join(dfile_path, 'img_trees_labeled-' + t_type + '.pickle'), 'wb') as handle:
        for img in images:
            pickle.dump(img, handle, protocol=pickle.HIGHEST_PROTOCOL)

    agg_dict = {}
    for i in hist_l.keys():
        if(hist_l[i] in agg_dict): agg_dict[hist_l[i]].append(i)
        else: agg_dict[hist_l[i]] = [i] 

    agg_dict = collections.OrderedDict(sorted(agg_dict.items(), key=lambda t: t[0]))

    mask_count = 0
    with open(path.join(dfile_path, 'codeword-count-{}.dict'.format(t_type)), 'w') as f:
        for de in agg_dict:
            mask_count += de*len(agg_dict[de])
            f_string = str(de) + ': '
            f_string += ' '.join(str(dv) for dv in agg_dict[de]) + '\n'
            f.write(f_string)
    print(mask_count)

    distribuition_plot(agg_dict, dfile_path)

main()