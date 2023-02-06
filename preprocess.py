#creat pickles for network:
import pickle
import argparse
from torch_geometric.data import Data
import networkx as nx
import pandas as pd


def main():

    # Read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', default=2010, type=str, required=True, help='year')
    parser.add_argument('--type', default='unweighted', type=str, required=True, help='Weighted or unweighted network')
    args = parser.parse_args()

    # Load networks
    networks = pd.read_csv("/content/networks_"+ args.year +".csv")

    # Extract weighted or unweighted network
    if args.type == 'weighted':
        G = nx.Graph()
        edges = zip(
            networks['node_1'],
            networks['node_2'],
            networks['weighted']
        )
        G.add_weighted_edges_from(edges)
        # Do something with network here
        with open("./" + 'network_{}.p'.format(args.year), 'wb') as f:
            pickle.dump(G, f)       
    else:
        G = nx.Graph()
        edges = zip(
            networks[networks.unweighted == 1]['node_1'],
            networks[networks.unweighted == 1]['node_2']
        )
        G.add_edges_from(edges)
        # Do something with network here
        with open("./" + 'network_{}.p'.format(args.year), 'wb') as f:
            pickle.dump(G, f)


if __name__ == '__main__':
    main()
