import argparse
import pickle
import random

import numpy as np
import torch
from helpers import get_best
from model import SparseSubspaceGAE


def main():

    # Read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs.')
    parser.add_argument('--cuda', type=int, default=0, help='Device to train.')
    parser.add_argument('--year', type=int, default=2013, help='Year for which to train model.')
    parser.add_argument('--random_seed', type=int, help='Random seed.')
    parser.add_argument('--lr', type=float, default=1e-04, help='Learning rate.')
    parser.add_argument('--lambda_r', type=float, default=1e-02, help='Regularization constant.')
    parser.add_argument('--lambda_o', type=float, default=1e-02, help='Orthogonalization constant.')
    parser.add_argument('--theta_d', type=int, default=768, help='Sparsity threshold.')
    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    year = args.year

    # Define path to files and filename
    filepath_data = 'data/final/'
    filepath_model = 'src/model/'
    filename = '{}'.format(year)
    if not args.lambda_o:
        filename += '_ra'
    if not args.lambda_r:
        filename += '_sa'

    # Load data
    with open(filepath_data + 'data_{}.p'.format(year), 'rb') as f:
        data = pickle.load(f)

    input_dim = 768
    hidden_dim = 10
    output_dim = 10

    # Initialize model
    model = SparseSubspaceGAE(
        input_dim,
        hidden_dim,
        output_dim,
        args.lr,
        args.lambda_r,
        args.lambda_o,
    )

    # Define device
    device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() else 'cpu')

    # Move model to device
    model = model.to(device)

    # Prepare data and move to device
    train_x_list = data.train_x_list
    dev_x_list = data.dev_x_list
    test_x_list = data.test_x_list
    train_pos_edge_index = data.train_pos_edge_index.to(device)
    dev_pos_edge_index = data.val_pos_edge_index
    dev_neg_edge_index = data.val_neg_edge_index
    test_pos_edge_index = data.test_pos_edge_index
    test_neg_edge_index = data.test_neg_edge_index

    print('Year {}, learning rate {:.0e}, lambda_r {:.0e}, lambda_o {:.0e}, sparsity {}...'.format(
        year, args.lr, args.lambda_r, args.lambda_o, args.theta_d))

    best_auc, _, _ = get_best(filepath_model + 'results/{}.txt'.format(filename), args.theta_d)
    print('Best AUC so far: {}'.format(best_auc))

    for epoch in range(1, args.epochs + 1):

        random.shuffle(train_x_list)
        for i, x in enumerate(train_x_list):
            x = x.to(device)
            model.train(x, train_pos_edge_index)
            if (i + 1) % 10 == 0:
                model.step()

        aucs = list()
        aps = list()
        for x in dev_x_list:
            x = x.to(device)
            auc, ap, _, _ = model.test(x, train_pos_edge_index, dev_pos_edge_index, dev_neg_edge_index)
            aucs.append(auc)
            aps.append(ap)
        auc_dev, ap_dev = np.mean(aucs), np.mean(aps)

        aucs = list()
        aps = list()
        for x in test_x_list:
            x = x.to(device)
            auc, ap, _, _ = model.test(x, train_pos_edge_index, test_pos_edge_index, test_neg_edge_index)
            aucs.append(auc)
            aps.append(ap)
        auc_test, ap_test = np.mean(aucs), np.mean(aps)

        sparse_weight = model.model.gae.encoder.conv1.weight
        not_pruned = len([r for r in sparse_weight if not torch.all(r == 0)])

        with open(filepath_model + 'results/{}.txt'.format(filename), 'a+') as f:
            f.write('{:.0e}\t{:.0e}\t{:.0e}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{}\n'.format(
                args.lr, args.lambda_r, args.lambda_o, auc_dev, ap_dev, auc_test, ap_test, not_pruned
            ))
        if best_auc is None or auc_dev > best_auc:
            if not_pruned <= args.theta_d:
                best_auc = auc_dev
                torch.save(model.state_dict(), filepath_model + 'trained/{}.torch'.format(filename))


if __name__ == '__main__':
    main()
