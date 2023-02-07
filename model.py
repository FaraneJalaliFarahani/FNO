import torch
from proxssi.groups.gcn import gcn_groups
from proxssi.optimizers.adamw_hf import AdamW
from sklearn.metrics import roc_auc_score, average_precision_score
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GAE, GCNConv


class MyGAE(GAE):
    def test(self, z, pos_edge_index, neg_edge_index):
        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        return roc_auc_score(y, pred), average_precision_score(y, pred), y, pred


class GCNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)


class SubspaceGAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, lambda_o):
        super(SubspaceGAE, self).__init__()
        self.rotate = lambda_o
        if self.rotate:
            self.linear = nn.Linear(input_dim, input_dim, bias=False)
        self.gae = MyGAE(GCNEncoder(input_dim, hidden_dim, output_dim))

    def forward(self, x, train_pos_edge_index):
        if self.rotate:
            x = self.linear(x)
        z = self.gae.encode(x, train_pos_edge_index)
        return z


class SparseSubspaceGAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, lr, lambda_r, lambda_o):
        super(SparseSubspaceGAE, self).__init__()
        self.model = SubspaceGAE(input_dim, hidden_dim, output_dim, lambda_o)
        grouped_params = gcn_groups(self.model, weight_decay=0)
        optimizer_kwargs = {
            'lr': lr,
            'penalty': 'l1_l2',
            'prox_kwargs': {'lambda_': lambda_r}
        }
        self.optimizer = AdamW(grouped_params, **optimizer_kwargs)
        self.optimizer.zero_grad()
        self.lambda_o = lambda_o
        self.input_dim = input_dim
        self.pruned = set()

    def train(self, x, train_pos_edge_index):
        self.model.train()
        z = self.model(x, train_pos_edge_index)
        loss = self.model.gae.recon_loss(z, train_pos_edge_index)
        if self.lambda_o:
            eye_matrix = torch.eye(self.input_dim).to(self.model.linear.weight.device)
            loss_o = torch.norm(self.model.linear.weight @ self.model.linear.weight.T - eye_matrix) ** 2
            loss += self.lambda_o * loss_o
        loss.backward()

    def step(self):
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.pruned.update([i for i, r in enumerate(self.model.gae.encoder.conv1.weight) if torch.all(r == 0)])
        if len(self.pruned) > 0:
            with torch.no_grad():
                self.model.gae.encoder.conv1.weight[torch.tensor(list(self.pruned))] = 0

    def test(self, x, train_pos_edge_index, test_pos_edge_index, test_neg_edge_index):
        self.model.eval()
        with torch.no_grad():
            z = self.model(x, train_pos_edge_index)
            auc, ap, y, pred = self.model.gae.test(z, test_pos_edge_index, test_neg_edge_index)
            return auc, ap, y, pred

    def get_embs(self, x):
        sparse_weight = self.model.gae.encoder.conv1.weight
        idxes_not_pruned = [i for i, r in enumerate(sparse_weight) if not torch.all(r == 0)]
        self.model.eval()
        with torch.no_grad():
            h = self.model.linear(x)
            return h[:, idxes_not_pruned]
