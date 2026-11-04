import torch
from torch import Tensor
import pdb
from torch_cluster import knn_graph, radius, radius_graph
from torch_geometric.data import Data, Batch, HeteroData
from utils import pairwise_directions
from typing import Optional


def build_rigid_graph(
    X: Tensor, A: Tensor, S: Tensor, mask: Tensor, max_neighbors: int = 30, cutoff: Optional[float] = None
) -> tuple[Tensor, Tensor]:
    """
    Build a rigid graph for a single protein structure in PyG format.

    Args:
        X (Tensor): Node coordinates of shape [M, sel, 3] where M is number of nodes,
                    sel is the number of atoms per residue, and the last axis is for
                    Cartesian coordinates.
        A (Tensor): Node atom type embedding of shape [M, sel, dim].
        S (Tensor): Residue sequence embedding of shape [M, dim].
        mask (Tensor): Mask tensor of shape [M] indicating valid residues.
        max_neighbors (int): Maximum number of neighbors for each node.

    Returns:
        edge_index (Tensor): Edge indices of shape [2, E] where E is number of edges.
        edge_attr (Tensor): Edge attributes of shape [E, edge_dim].
    """
    X = X[mask.long() > 0]
    M, sel = X.shape[0], X.shape[1]

    # Compute residue center of mass (or use alpha carbon position)
    # Using mean of all atom positions as representative position
    # X_ca = X.mean(dim=1)  # [M, 3]
    X = X.view(M * sel, 3)  # [M * sel, 3]

    if cutoff is not None:
        # Build radius graph based on residue positions
        edge_index = radius_graph(X, r=cutoff, max_num_neighbors=max_neighbors, loop=False)
    else:
        # Build k-NN graph based on residue positions
        edge_index = knn_graph(X, k=max_neighbors, loop=False)

    # Compute edge attributes (relative positions and distances)
    row, col = edge_index
    edge_vec = X[col] - X[row]  # [E, 3]
    edge_dist = edge_vec.norm(dim=-1, keepdim=True)  # [E, 1]

    # Normalize edge vectors
    # edge_vec_norm = edge_vec / (edge_dist + 1e-8)

    # Combine edge attributes
    # edge_attr = torch.cat([edge_vec, edge_dist, edge_vec_norm], dim=-1)  # [E, 7]
    # direction_features = pairwise_directions(X.view(M, sel, 3)[None, ...]) # [1, M*sel, M*sel, 12]
    # edge_attr  = torch.cat([edge_dist, direction_features[0, row, col]], dim=-1)  # [E, 1 + 12 = 13]
    edge_attr = edge_dist  # [E, 1]

    return edge_index, edge_attr


def build_rigid_graph_batch(
    X: Tensor, H: Tensor, A: Tensor, S: Tensor, mask: Tensor, max_neighbors: int = 30, cutoff: Optional[float] = None
) -> Data:
    """
    Build a rigid graph batch for a batch of protein structures in PyG format.

    Args:
        X (Tensor): Node coordinates of shape [B, M, sel, 3].
        H (Tensor): Node embedding of shape [B, M, sel, dim_h].
        A (Tensor): Node atom type embedding of shape [B, M, sel, dim].
        mask (Tensor): Mask tensor of shape [B, M] indicating valid residues.
        S (Tensor): Residue sequence embedding of shape [B, M, sel, dim].
        max_neighbors (int): Maximum number of neighbors for each node.

    Returns:
        Data: PyG Data object containing the batched graph.
    """
    B, _, sel, _ = X.shape
    _, _, _, atom_dim = A.shape
    _, _, _, seq_dim = S.shape

    # Lists to collect graph components
    edge_index_list = []
    edge_attr_list = []
    x_list = []
    pos_list = []
    batch_list = []

    for b in range(B):
        # Build graph for each protein in the batch
        edge_index, edge_attr = build_rigid_graph(
            X[b], A[b], S[b], mask[b], max_neighbors=max_neighbors, cutoff=cutoff
        )

        M = mask[b].long().sum().item()

        # Offset edge indices for batching
        if b > 0:
            edge_index = edge_index + M * sel

        # Flatten atom embeddings and combine with sequence embeddings
        # A_flat = A[b].reshape(M * sel, -1)  # [M * sel, atom_dim]
        # node_features = torch.cat(
        #     [A_flat, S[b]], dim=-1
        # )  # [M, sel * atom_dim + seq_dim]
        node_features = H[b][mask[b].long() > 0].view(M * sel, -1)  # [M * sel, h_dim]

        # Use center of mass as node positions
        # nonzero_mask = (X[b].abs().sum(dim=-1) != 0).float()  # [M, sel]
        # pos = (X[b] * nonzero_mask.unsqueeze(-1)).sum(dim=1) / (
        #     nonzero_mask.sum(dim=1, keepdim=True) + 1e-8
        # )  # [M, 3]
        pos = X[b][mask[b].long() > 0].view(M * sel, 3) # [M * sel, 3]

        # Create batch tensor
        batch = torch.full((M * sel,), b, dtype=torch.long)

        edge_index_list.append(edge_index)
        edge_attr_list.append(edge_attr)
        x_list.append(node_features)
        pos_list.append(pos)
        batch_list.append(batch)

    # Concatenate all components
    edge_index = torch.cat(edge_index_list, dim=1)
    edge_attr = torch.cat(edge_attr_list, dim=0)
    x = torch.cat(x_list, dim=0)
    pos = torch.cat(pos_list, dim=0)
    batch = torch.cat(batch_list, dim=0)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=pos,
        batch=batch,
        # X_atoms=X_atoms,  # Store original atom coordinates
        # A_atoms=A_atoms,  # Store original atom embeddings
        # num_nodes=B * M * sel,
    )

def build_cross_graph(
    bind_X: Tensor,
    bind_H: Tensor,
    bind_A: Tensor,
    bind_S: Tensor,
    tgt_X: Tensor,
    tgt_H: Tensor,
    tgt_A: Tensor,
    tgt_S: Tensor,
    bind_mask: Tensor,
    tgt_mask: Tensor,
    cutoff: float = 15.0,
    batch_idx: int = 0
) -> HeteroData:
    """
    Build a cross-graph between binder and target proteins for docking.

    Args:
        bind_X (Tensor): Binder coordinates of shape [M1, sel, 3].
        bind_H (Tensor): Binder node embedding of shape [M1, sel, dim_h].
        bind_A (Tensor): Binder atom type embedding of shape [M1, sel, dim].
        bind_S (Tensor): Binder sequence embedding of shape [M1, dim].
        tgt_X (Tensor): Target coordinates of shape [M2, sel, 3].
        tgt_H (Tensor): Target node embedding of shape [M2, sel, dim_h].
        tgt_A (Tensor): Target atom type embedding of shape [M2, sel, dim].
        tgt_S (Tensor): Target sequence embedding of shape [M2, dim].
        cutoff (float): Distance cutoff for cross edges.

    Returns:
        HeteroData: Heterogeneous graph with binder and target nodes and their interactions.
    """
    bind_H = bind_H[bind_mask.long() > 0]
    tgt_H = tgt_H[tgt_mask.long() > 0]

    # Create heterogeneous data object
    data = HeteroData()

    # Compute representative positions (center of mass)
    # bind_nonzero_mask = (bind_X.abs().sum(dim=-1) != 0).float()  # [M1, sel1]
    # bind_pos = (bind_X * bind_nonzero_mask.unsqueeze(-1)).sum(dim=1) / (
    #     bind_nonzero_mask.sum(dim=1, keepdim=True) + 1e-8
    # )  # [M1, 3]
    # tgt_nonzero_mask = (tgt_X.abs().sum(dim=-1) != 0).float()  # [M2, sel2]
    # tgt_pos = (tgt_X * tgt_nonzero_mask.unsqueeze(-1)).sum(dim=1) / (
    #     tgt_nonzero_mask.sum(dim=1, keepdim=True) + 1e-8
    # )  # [M2, 3]

    # Build intra-protein graphs
    bind_edge_index, bind_edge_attr = build_rigid_graph(bind_X, bind_A, bind_S, bind_mask)
    tgt_edge_index, tgt_edge_attr = build_rigid_graph(tgt_X, tgt_A, tgt_S, tgt_mask)

    bind_X = bind_X[bind_mask.long() > 0]
    tgt_X = tgt_X[tgt_mask.long() > 0]

    M1, sel1, _ = bind_X.shape
    M2, sel2, _ = tgt_X.shape

    # Store binder node features
    data["binder"].x = bind_H.view(M1 * sel1, -1)
    # data["binder"].pos = bind_X.view(M1 * sel1, 3)
    # data["binder"].X_atoms = bind_X.view(M1 * sel1, 3)
    data["binder", "intra", "binder"].edge_index = bind_edge_index
    data["binder", "intra", "binder"].edge_attr = bind_edge_attr

    # Store target node features
    data["target"].x = tgt_H.view(M2 * sel2, -1)
    # data["target"].pos = tgt_X
    # data["target"].X_atoms = tgt_X
    data["target", "intra", "target"].edge_index = tgt_edge_index
    data["target", "intra", "target"].edge_attr = tgt_edge_attr


    bind_X_flat = bind_X.view(M1 * sel1, 3)
    tgt_X_flat = tgt_X.view(M2 * sel2, 3)

    bind_batch = torch.zeros(M1 * sel1, dtype=torch.long).to(bind_X_flat.device)
    tgt_batch = torch.zeros(M2 * sel2, dtype=torch.long).to(tgt_X_flat.device)

    bind_batch[:] = batch_idx
    tgt_batch[:] = batch_idx

    edge_index = radius(
        x=tgt_X_flat,
        y=bind_X_flat,
        r=cutoff,
        batch_x=tgt_batch,
        batch_y=bind_batch,
        max_num_neighbors=1000,
    )

    src, dst = edge_index # src=target, dst=binder
    edge_dist = (bind_X_flat[src] - tgt_X_flat[dst]).norm(dim=-1, keepdim=True)  # [E, 1]
    edge_attr = edge_dist  # [E, 1]

    data["target", "to", "binder"].edge_index = edge_index
    data["target", "to", "binder"].edge_attr = edge_attr
    data["binder", "to", "target"].edge_index = edge_index.flip(0)
    data["binder", "to", "target"].edge_attr = edge_attr
    return data


def build_cross_graph_batch(
    bind_X: Tensor,
    bind_H: Tensor,
    bind_A: Tensor,
    bind_S: Tensor,
    tgt_X: Tensor,
    tgt_H: Tensor,
    tgt_A: Tensor,
    tgt_S: Tensor,
    cutoffs: Tensor,
    bind_mask: Tensor,
    tgt_mask: Tensor,
) -> Batch:
    """
    Build a batch of cross-graphs between binder and target proteins for docking.

    Args:
        bind_X (Tensor): Binder coordinates of shape [B, M1, sel, 3].
        bind_H (Tensor): Binder node embedding of shape [B, M1, sel, dim_h].
        bind_A (Tensor): Binder atom type embedding of shape [B, M1, sel, dim].
        bind_S (Tensor): Binder sequence embedding of shape [B, M1, sel, dim].
        tgt_X (Tensor): Target coordinates of shape [B, M2, sel, 3].
        tgt_H (Tensor): Target node embedding of shape [B, M2, sel, dim_h].
        tgt_A (Tensor): Target atom type embedding of shape [B, M2, sel, dim].
        tgt_S (Tensor): Target sequence embedding of shape [B, M2, sel, dim].
        cutoff (Tensor): Distance cutoffs for cross edges.
        bind_mask (Tensor): Mask tensor of shape [B, M1] indicating valid binder residues.
        tgt_mask (Tensor): Mask tensor of shape [B, M2] indicating valid target residues.

    Returns:
        Batch: Batched heterogeneous graph with binder and target nodes and their interactions.
    """
    B = bind_X.shape[0]

    # List to collect individual cross graphs
    graph_list = []

    if isinstance(cutoffs, (int, float)):
        cutoffs = torch.full((B,), cutoffs, dtype=torch.float)

    bind_X_all = []
    tgt_X_all = []
    bind_batch_all = []
    tgt_batch_all = []

    for b, cutoff in zip(range(B), cutoffs):
        # Build cross graph for each protein pair
        bind_X_ = bind_X[b][bind_mask[b].long() > 0]
        tgt_X_ = tgt_X[b][tgt_mask[b].long() > 0]

        M1, sel1, _ = bind_X_.shape
        M2, sel2, _ = tgt_X_.shape

        bind_X_flat = bind_X_.view(M1 * sel1, 3)
        tgt_X_flat = tgt_X_.view(M2 * sel2, 3)

        bind_batch = torch.zeros(M1 * sel1, dtype=torch.long).to(bind_X_flat.device)
        tgt_batch = torch.zeros(M2 * sel2, dtype=torch.long).to(tgt_X_flat.device)

        bind_batch[:] = b
        tgt_batch[:] = b

        bind_X_all.append(bind_X_flat)
        tgt_X_all.append(tgt_X_flat)
        bind_batch_all.append(bind_batch)
        tgt_batch_all.append(tgt_batch)

    bind_X_all = torch.cat(bind_X_all, dim=0)
    tgt_X_all = torch.cat(tgt_X_all, dim=0)
    bind_batch_all = torch.cat(bind_batch_all, dim=0)
    tgt_batch_all = torch.cat(tgt_batch_all, dim=0)

    edge_index = radius(
        x=tgt_X_all,
        y=bind_X_all,
        r=cutoff,
        batch_x=tgt_batch_all,
        batch_y=bind_batch_all,
        max_num_neighbors=1000,
    )

    src, dst = edge_index # src=binder, dst=target
    edge_dist = (tgt_X_all[dst] - bind_X_all[src]).norm(dim=-1, keepdim=True)  # [E, 1]
    edge_attr = edge_dist  # [E, 1]

    return edge_index, edge_attr, bind_batch_all, tgt_batch_all
