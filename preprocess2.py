import argparse
import pickle
from collections import defaultdict

import networkx as nx
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.utils import train_test_split_edges


def main():

    # Read year and window size
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, help='Year for which to create data splits.')
    args = parser.parse_args()

    # Define paths
    inpath = ''
    outpath = ''

    year = args.year

    print('Processing {}...'.format(year))

    # Load concepts, concept embeddings, concept counts, and network
    with open(inpath + 'network_{}.p'.format(year), 'rb') as f:
        G = pickle.load(f)
    # with open(inpath + 'concepts/concepts_{}.txt'.format(year), 'r') as f:
    #     concepts = set(f.read().strip().split('\n'))

    # with open(inpath + 'embeddings/embeddings_{}.p'.format(year), 'rb') as f:
    #     sr_c_embs = pickle.load(f)
    concepts = ['concept1', 'concept2', 'concept3', 'concept4', 'concept5']
    c_sr_embs = torch.rand(len(concepts), G.number_of_nodes(), 768)


    # Create dictionary mapping concepts to subreddits
    c_sr_embs = defaultdict(dict)
    #for sr in G.nodes():
    #    for c in sr_c_embs[sr]:
    #        if c in concepts:
    #            c_sr_embs[c][sr] = sr_c_embs[sr][c]

    # Store number of subreddits
    n_srs = G.number_of_nodes()

    # Define look-up dictionaries for concepts and subreddits
    id2concept = {i: concept for i, concept in concepts)}
    id2sr = {i: sr for i, sr in enumerate(G.nodes())}

    # Compute embedding matrix for all concepts
    x_list = list()
    for i in id2concept:
        c = id2concept[i]
        x = torch.zeros(n_srs, 768)
        for j in id2sr:
            sr = id2sr[j]
            print(sr)
            print(c_sr_embs[c])
            if sr in c_sr_embs[c]:
                print("not-NAN")
           
                x[j] = torch.tensor(c_sr_embs[c][sr])
            else:
                print("NAN")
                x[j] = np.nan
        nan_mask = torch.isnan(x)
        mean_x = torch.tensor(np.nanmean(x, axis=0)).repeat(x.size(0), 1)
        x[nan_mask] = mean_x[nan_mask]
        x_list.append(x)

    # Split into train, dev, and test
    train_x_list, dev_test_x_list = train_test_split(x_list, test_size=0.4, shuffle=True, random_state=123)
    dev_x_list, test_x_list = train_test_split(dev_test_x_list, test_size=0.5, shuffle=True, random_state=123)

    # Define edge list
    A = nx.adjacency_matrix(G, nodelist=G.nodes())
    edge_index = torch.tensor(np.stack((A.tocoo().row, A.tocoo().col)).astype(np.int32), dtype=torch.long)

    # Define dataset and split into train, dev, and test
    data = Data(
        train_x_list=train_x_list,
        dev_x_list=dev_x_list,
        test_x_list=test_x_list,
        edge_index=edge_index,
        id2concept=id2concept,
        id2sr=id2sr
    )
    data_split = train_test_split_edges(data, val_ratio=0.2, test_ratio=0.2)

    with open(outpath + 'data_{}.p'.format(year), 'wb') as f:
        pickle.dump(data_split, f)

    print('{} processed.'.format(year))


if __name__ == '__main__':
    main()
