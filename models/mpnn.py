import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph import build_cross_graph_batch, build_rigid_graph_batch
from torch import Tensor
from typing import List, Optional, Union
from torch_geometric.nn import TransformerConv, GATConv, GATv2Conv
from torch_scatter import scatter
from torch_geometric.utils import to_dense_batch
from .modules import GraphGaussianSmearing

class MPNN(torch.nn.Module):
    def __init__(self, args):
        super(MPNN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.fc = nn.Sequential(
            nn.Linear(args.hidden_size + args.noise_emb_dim, args.hidden_size),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.hidden_size, args.hidden_size),
        )
        for idx in range(1):
            i_dim = args.hidden_size
            if idx == 0:
                o_dim = args.hidden_size

            self.convs.append(
                GATv2Conv(
                    i_dim, o_dim,
                    heads=args.heads,
                    concat=False,
                    dropout=args.dropout,
                    edge_dim=args.hidden_size,
                    share_weights=True
                )
            )

        self.distance_mlp = nn.Sequential(
            nn.Linear(args.hidden_size // 2, args.hidden_size),
            # nn.ReLU(),
            # nn.Dropout(args.dropout),
            # nn.Linear(args.hidden_size // 2, args.hidden_size),
        )

    def forward(
        self,
        x: Union[Tensor, List[Tensor]],
        edge_index: Tensor,
        edge_attr: Tensor,
        time_emb: Tensor,
    ) -> Tensor:
        # edge_attr = self.distance_embedding(edge_attr).squeeze(0).squeeze(0)
        time_emb = time_emb.unsqueeze(0).expand(x.size(0), -1)
        x = self.fc(torch.cat((x, time_emb), dim=-1))

        edge_attr = self.distance_mlp(edge_attr)
        for idx, conv in enumerate(self.convs):
            h = conv(x, edge_index, edge_attr)
            x = h + x

        return x


class CrossMPNN(nn.Module):
    def __init__(self, args):
        super(CrossMPNN, self).__init__()
        self.in_dim = args.hidden_size
        self.out_dim = args.hidden_size
        self.depth = 1
        self.layers = nn.ModuleList()

        for _ in range(self.depth):
            layer = CrossMPNNLayer(args)
            self.layers.append(layer)

        self.distance_mlp = nn.Sequential(
            nn.Linear(args.hidden_size // 2, args.hidden_size),
            # nn.ReLU(),
            # nn.Dropout(args.dropout),
            # nn.Linear(args.hidden_size // 2, args.hidden_size),
        )

    def forward(
        self,
        node_attr,
        edge_index,
        edge_attr,
        time_emb,
        out_nodes=None,
        reduction="mean",
        debug=False,
    ):
        """
        @param node_attr  [N, D]
        @param edge_index  [2, E]
        @param edge_attr  [E, D]
        @param out_nodes  int
        """
        # edge_attr = self.distance_embedding(edge_attr).squeeze(0).squeeze(0)
        edge_attr = self.distance_mlp(edge_attr)

        for layer in self.layers:
            node_attr = layer(
                node_attr,
                edge_index,
                edge_attr,
                time_emb,
                out_nodes=out_nodes,
                reduction=reduction,
                debug=debug,
            )

        return node_attr


class CrossMPNNLayer(nn.Module):
    def __init__(self, args):
        super(CrossMPNNLayer, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(args.hidden_size + args.noise_emb_dim, args.hidden_size),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.hidden_size, args.hidden_size),
        )

        self.batch_norm = nn.BatchNorm1d(args.hidden_size)

    def forward(
        self,
        node_attr,
        edge_index,
        edge_attr,
        time_emb,
        out_nodes=None,
        reduction="mean",
        debug=False,
    ):
        """
        @param node_attr  [N, D]
        @param edge_index  [2, E]
        @param edge_attr  [E, D]
        @param out_nodes  int
        """
        edge_src, edge_dst = edge_index
        # prod = node_attr[edge_src] * self.fc(node_attr[edge_dst]) * edge_attr
        if debug:
            pdb.set_trace()

        time_emb = time_emb.unsqueeze(0).expand(node_attr.size(0), -1)
        node_attr = self.fc(torch.cat((node_attr, time_emb), dim=-1))
        # pdb.set_trace()
        prod = node_attr[edge_dst] * edge_attr

        out_nodes = out_nodes or node_attr.shape[0]

        if debug:
            pdb.set_trace()

        out = scatter(prod, edge_src, dim=0, dim_size=out_nodes, reduce=reduction)

        # out = out + node_attr
        # out = self.batch_norm(out)

        return out


class MPNNEncoder(nn.Module):
    def __init__(self, args):
        super(MPNNEncoder, self).__init__()
        self.depth = args.depth
        self.inter_layers = nn.ModuleList()
        self.intra_layers = nn.ModuleList()
        # self.distance_embedding = DistanceEmbedding(args.hidden_size // 2)
        self.distance_embedding = GraphGaussianSmearing(0, 30, args.hidden_size // 2)

        for _ in range(self.depth):
            self.inter_layers.append(CrossMPNN(args))
            self.intra_layers.append(MPNN(args))
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, bind_X, bind_H, bind_A, bind_S, tgt_X, tgt_H, tgt_A, tgt_S, n_neighbors, cutoff, time_emb, bind_mask, tgt_mask):
        bind_graph = build_rigid_graph_batch(
            bind_X, bind_H, bind_A, bind_S, bind_mask, n_neighbors
        )
        tgt_graph = build_rigid_graph_batch(
            tgt_X, tgt_H, tgt_A, tgt_S, tgt_mask, n_neighbors
        )
        cross_edge_index, cross_edge_attr, bind_batch, tgt_batch = build_cross_graph_batch(
            bind_X, bind_H, bind_A, bind_S, tgt_X, tgt_H, tgt_A, tgt_S, cutoff, bind_mask, tgt_mask
        )

        # cross_dist_emb = self.distance_embedding(cross_graph[('target', 'to', 'binder')]['edge_attr'])
        cross_dist_emb = self.distance_embedding(cross_edge_attr)
        bind_dist_emb = self.distance_embedding(bind_graph.edge_attr)
        tgt_dist_emb = self.distance_embedding(tgt_graph.edge_attr)

        B = bind_H.size()[0]
        D = bind_H.size(-1)

        bind_H_all = []
        tgt_H_all = []

        for b in range(B):
            bind_H_all.append(bind_H[b][bind_mask[b] > 0])
            tgt_H_all.append(tgt_H[b][tgt_mask[b] > 0])

        # bind_H = bind_H.view(-1, bind_H.size(-1))
        # tgt_H = tgt_H.view(-1, tgt_H.size(-1))

        bind_H = torch.cat(bind_H_all, dim=0).view(-1, bind_H.size(-1)) # (sum(N_b), D)
        tgt_H = torch.cat(tgt_H_all, dim=0).view(-1, tgt_H.size(-1)) # (sum(N_t), D)

        for i in range(self.depth):
            bind_H_intra = self.intra_layers[i](bind_H, bind_graph.edge_index, bind_dist_emb, time_emb) # ligand conv

            cross_bind_H = self.inter_layers[i](
                tgt_H,
                cross_edge_index,
                # cross_graph[('target', 'to', 'binder')]['edge_index'],
                cross_dist_emb,
                time_emb,
                out_nodes=bind_H.shape[0]
                # cross_graph['binder']['x'].shape[0]
                # out_nodes=bind_graph.x.size(0),
            )

            tgt_H_intra = self.intra_layers[i](tgt_H, tgt_graph.edge_index, tgt_dist_emb, time_emb) # receptor conv
            cross_tgt_H = self.inter_layers[i](
                bind_H,
                cross_edge_index.flip(0),
                # cross_graph[('binder', 'to', 'target')]['edge_index'],
                cross_dist_emb,
                time_emb,
                out_nodes=tgt_H.shape[0]
                # out_nodes=tgt_graph.x.size(0),
            )

            bind_H = bind_H + bind_H_intra + cross_bind_H
            tgt_H = tgt_H +  tgt_H_intra + cross_tgt_H

        bind_H = to_dense_batch(bind_H, bind_batch)[0]
        tgt_H = to_dense_batch(tgt_H, tgt_batch)[0]

        return bind_H.view(B, -1, D), tgt_H.view(B, -1, D)
