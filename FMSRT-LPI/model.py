import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_sparse import SparseTensor
from torch.utils.data import DataLoader
from torch_geometric.nn import Linear, GINConv
from torch_geometric.utils import add_self_loops, negative_sampling, degree
from sklearn.metrics import roc_auc_score, average_precision_score
from utils import calculate_metrics

class PolyLoss(nn.Module):
    def __init__(self, DEVICE,weight_loss=None, epsilon=1.0):
        super(PolyLoss, self).__init__()
        self.CELoss = nn.CrossEntropyLoss(weight=weight_loss, reduction='none')
        self.epsilon = epsilon
        self.DEVICE = DEVICE

    def forward(self, predicted, labels):
        batch_size=labels.shape[0]
        one_hot = torch.zeros((batch_size, 2), device=self.DEVICE).scatter_(
            1, labels.to(torch.int64), 1)
        pt = torch.sum(one_hot * F.softmax(predicted, dim=1), dim=-1)
        ce = self.CELoss(predicted, labels.to(torch.float32))
        poly1 = ce + self.epsilon * (1-pt)
        return torch.mean(poly1)

class att_layer(torch.nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim):
        super(att_layer, self).__init__()
        self.scale = 128 ** -0.5

        self.q = torch.nn.Linear(input_dim, hid_dim)
        self.k = torch.nn.Linear(input_dim, hid_dim)
        self.v = torch.nn.Linear(input_dim, hid_dim)

        self.out_layer = torch.nn.Linear(hid_dim, output_dim)

    def forward(self, q, k, v):
        q = self.q(q)
        k = self.k(k)
        v = self.v(v)

        attention = torch.mm(q, k.t()) * self.scale
        attention = F.softmax(attention, dim=-1)
        out = torch.mm(attention.float(), v)
        return self.out_layer(out)


class GNNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNNEncoder, self).__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(GINConv(Linear(in_channels, hidden_channels), train_eps=True))
        self.convs.append(GINConv(Linear(hidden_channels, out_channels), train_eps=True))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.bns.append(nn.BatchNorm1d(out_channels))

        self.dropout = nn.Dropout(0.5)
        self.activation = nn.ELU()

        self.attention = att_layer(out_channels, out_channels, out_channels)

    def forward(self, x, edge_index):
        edge_index = SparseTensor.from_edge_index(edge_index, sparse_sizes=(x.size(0), x.size(0))).cuda()
        for i, conv in enumerate(self.convs[:-1]):
            x = self.dropout(x)
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = self.activation(x)
        x = self.dropout(x)
        x = self.convs[-1](x, edge_index)
        x = self.bns[-1](x)
        x = self.activation(x)
        # x = self.attention(x, x, x)
        return x


class EdgeDecoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(EdgeDecoder, self).__init__()
        self.mlps = nn.ModuleList()
        self.mlps.append(nn.Linear(in_channels, hidden_channels))
        self.mlps.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = nn.Dropout(0.5)
        self.activation = nn.ELU()

        self.attention = att_layer(out_channels, out_channels, out_channels)

    def forward(self, z, edge):
        x = z[edge[0]] * z[edge[1]]
        for i, mlp in enumerate(self.mlps[:-1]):
            x = self.dropout(x)
            x = mlp(x)
            x = self.activation(x)
        x = self.mlps[-1](x)
        # att_x = self.attention(x, x, x)
        # x = att_x + x
        return x


class DegreeDecoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=1):
        super(DegreeDecoder, self).__init__()
        self.mlps = nn.ModuleList()
        self.mlps.append(nn.Linear(in_channels, hidden_channels))
        self.mlps.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = nn.Dropout(0.5)
        self.activation = nn.ELU()

        self.attention = att_layer(out_channels, out_channels, out_channels)

    def forward(self, x):
        for i, mlp in enumerate(self.mlps[:-1]):
            x = mlp(x)
            x = self.dropout(x)
            x = self.activation(x)
        x = self.mlps[-1](x)
        x = self.activation(x)
        # att_x = self.attention(x, x, x)
        # x = att_x + x
        return x


def ce_loss(pos_out, neg_out):
    pos_loss = F.binary_cross_entropy(pos_out.sigmoid(), torch.ones_like(pos_out))
    neg_loss = F.binary_cross_entropy(neg_out.sigmoid(), torch.zeros_like(neg_out))
    
    return pos_loss + neg_loss


class GMAE(nn.Module):
    def __init__(self, encoder, edge_decoder, degree_decoder, mask):
        super(GMAE, self).__init__()
        self.encoder = encoder
        self.edge_decoder = edge_decoder
        self.degree_decoder = degree_decoder
        self.mask = mask
        self.negative_sampler = negative_sampling
        self.poly_loss=PolyLoss(DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
    def loss_fn(self,pos_out, neg_out):
        pos_one_hot=torch.tensor([[0,1]]).repeat(len(pos_out),1).cuda()
        neg_one_hot=torch.tensor([[1,0]]).repeat(len(neg_out),1).cuda()
        pos_loss = self.poly_loss(pos_out, pos_one_hot)
        neg_loss = self.poly_loss(neg_out, neg_one_hot)
        return pos_loss + neg_loss
    
    def train_epoch(self, data, optimizer, alpha, batch_size=8192, grad_norm=1.0):
        x, edge_index = data.x, data.edge_index
        remaining_edges, masked_edges = self.mask(edge_index)
        aug_edge_index, _ = add_self_loops(edge_index)
        neg_edges = self.negative_sampler(
            aug_edge_index, num_nodes=data.num_nodes, num_neg_samples=masked_edges.view(2, -1).size(1)
        ).view_as(masked_edges)
        for perm in DataLoader(range(masked_edges.size(1)), batch_size=batch_size, shuffle=True):
            optimizer.zero_grad()
            z = self.encoder(x, remaining_edges)

            batch_masked_edges = masked_edges[:, perm]
            batch_neg_edges = neg_edges[:, perm]
            pos_out = self.edge_decoder(z, batch_masked_edges)
            neg_out = self.edge_decoder(z, batch_neg_edges)
            loss = self.loss_fn(pos_out, neg_out)

            deg = degree(masked_edges[1].flatten(), data.num_nodes).float()
            loss += alpha * F.mse_loss(self.degree_decoder(z).squeeze(), deg)

            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), grad_norm)
            optimizer.step()

    @torch.no_grad()
    def batch_predict(self, z, edges, batch_size=2 ** 16):
        preds = []
        for perm in DataLoader(range(edges.size(1)), batch_size):
            edge = edges[:, perm]
            preds += [self.edge_decoder(z, edge).squeeze().cpu()]
        pred = torch.cat(preds, dim=0)
        pred = F.softmax(pred,dim=1)[:,1]
        return pred

    @torch.no_grad()
    def test(self, z, pos_edge_index, neg_edge_index):
        pos_pred = self.batch_predict(z, pos_edge_index)
        neg_pred = self.batch_predict(z, neg_edge_index)

        pred = torch.cat([pos_pred, neg_pred], dim=0)
        pos_y = pos_pred.new_ones(pos_pred.size(0))
        neg_y = neg_pred.new_zeros(neg_pred.size(0))

        y = torch.cat([pos_y, neg_y], dim=0)
        y, pred = y.cpu().numpy(), pred.cpu().numpy()

        auc = roc_auc_score(y, pred)
        ap = average_precision_score(y, pred)
        temp = torch.tensor(pred)
        temp[temp >= 0.5] = 1
        temp[temp < 0.5] = 0
        acc, sen, pre, spe, F1, mcc = calculate_metrics(y, temp.cpu())
        return auc, ap, acc, sen, pre, spe, F1, mcc

    @torch.no_grad()
    def get_embedding(self, x, edge_index, mode="cat", l2_normalize=False):

        self.eval()
        assert mode in {"cat", "last"}, mode

        x = self.create_input_feat(x)
        edge_index = SparseTensor.from_edge_index(edge_index, sparse_sizes=(x.size(0), x.size(0))).cuda()
        out = []
        for i, conv in enumerate(self.convs[:-1]):
            x = self.dropout(x)
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = self.activation(x)
            out.append(x)
        x = self.dropout(x)
        x = self.convs[-1](x, edge_index)
        x = self.bns[-1](x)
        x = self.activation(x)
        out.append(x)

        if mode == "cat":
            embedding = torch.cat(out, dim=1)
        else:
            embedding = out[-1]

        if l2_normalize:
            embedding = F.normalize(embedding, p=2, dim=1)  # Cora, Citeseer, Pubmed

        return embedding

