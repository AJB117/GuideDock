import torch
import torch.nn as nn
import torch.nn.functional as F


class TriangleMultiplicationOutgoing(nn.Module):
    """
    pair[i, j] += sum_k W_left(pair[i, k]) * W_right(pair[k, j])
    """

    def __init__(self, c_pair, c_hidden):
        super().__init__()
        self.ln = nn.LayerNorm(c_pair)

        self.left = nn.Linear(c_pair, c_hidden, bias=False)
        self.right = nn.Linear(c_pair, c_hidden, bias=False)

        self.gate = nn.Linear(c_pair, c_hidden)
        self.out = nn.Linear(c_hidden, c_pair)

    def forward(self, pair):

        x = self.ln(pair)

        a = self.left(x)
        b = self.right(x)
        g = torch.sigmoid(self.gate(x))

        out = torch.einsum("bikh,bkjh->bijh", a, b)
        out = out * g

        return self.out(out)


class TriangleMultiplicationIncoming(nn.Module):
    """
    pair[i, j] += sum_k W_left(pair[k, i]) * W_right(pair[j, k])
    """

    def __init__(self, c_pair, c_hidden):
        super().__init__()
        self.ln = nn.LayerNorm(c_pair)

        self.left = nn.Linear(c_pair, c_hidden, bias=False)
        self.right = nn.Linear(c_pair, c_hidden, bias=False)

        self.gate = nn.Linear(c_pair, c_hidden)
        self.out = nn.Linear(c_hidden, c_pair)

    def forward(self, pair):

        x = self.ln(pair)

        a = self.left(x)
        b = self.right(x)
        g = torch.sigmoid(self.gate(x))

        out = torch.einsum("bkih,bjkh->bijh", a, b)
        out = out * g

        return self.out(out)


class PairTransition(nn.Module):
    def __init__(self, c_pair, expansion=4):
        super().__init__()
        self.ln = nn.LayerNorm(c_pair)
        self.fc1 = nn.Linear(c_pair, c_pair * expansion)
        self.fc2 = nn.Linear(c_pair * expansion, c_pair)

    def forward(self, pair):

        x = self.ln(pair)
        x = F.silu(self.fc1(x))
        return self.fc2(x)


class PairEmbeddingInit(nn.Module):
    def __init__(
        self,
        c_pair,
        num_aa=21,
        max_relpos=32,
        symmetric=True,
    ):
        super().__init__()
        self.c_pair = c_pair
        self.num_aa = num_aa
        self.max_relpos = max_relpos
        self.symmetric = symmetric

        self.relpos = nn.Embedding(2 * max_relpos + 1, c_pair)
        self.aa_pair = nn.Embedding(num_aa * num_aa, c_pair)
        self.ln = nn.LayerNorm(c_pair)

        nn.init.normal_(self.relpos.weight, std=0.02)
        nn.init.normal_(self.aa_pair.weight, std=0.02)

    def forward(self, aatype):

        B, L = aatype.shape
        device = aatype.device

        idx = torch.arange(L, device=device)
        rel = idx[:, None] - idx[None, :]
        rel = rel.clamp(-self.max_relpos, self.max_relpos) + self.max_relpos
        rel = self.relpos(rel)
        rel = rel.unsqueeze(0).expand(B, -1, -1, -1)

        aa_i = aatype[:, :, None]
        aa_j = aatype[:, None, :]
        aa_idx = aa_i * self.num_aa + aa_j
        aa = self.aa_pair(aa_idx)

        pair = rel + aa

        if self.symmetric:
            pair = 0.5 * (pair + pair.transpose(1, 2))

        return self.ln(pair)


class PairStack(nn.Module):
    """
    AF2-style pair stack using ONLY triangle multiplicative updates.
    """

    def __init__(
        self,
        c_pair,
        c_hidden,
        n_blocks,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "tri_out": TriangleMultiplicationOutgoing(c_pair, c_hidden),
                        "tri_in": TriangleMultiplicationIncoming(c_pair, c_hidden),
                        "transition": PairTransition(c_pair),
                    }
                )
                for _ in range(n_blocks)
            ]
        )

    def forward(self, pair):

        for blk in self.blocks:
            pair = pair + blk["tri_out"](pair)
            pair = pair + blk["tri_in"](pair)
            pair = pair + blk["transition"](pair)
        return pair


class TriangleModule(nn.Module):
    def __init__(self, c_pair=128, c_hidden=64, n_blocks=4):
        super().__init__()
        self.stack = PairStack(
            c_pair=c_pair,
            c_hidden=c_hidden,
            n_blocks=n_blocks,
        )

    def forward(self, pair):

        pair = self.stack(pair)
        return pair


if __name__ == "__main__":
    B, L = 2, 128
    aatype = torch.randint(0, 21, (B, L))
    model = TriangleModule(c_pair=128, c_hidden=64, n_blocks=3)
    pair = model(aatype)
    print(pair.shape)
