import torch
import torch.nn as nn


class MF(nn.Module):
    def __init__(self, n_user, n_item, dim):
        super(MF, self).__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.U_emb = nn.Embedding(num_embeddings=n_user, embedding_dim=dim)
        self.V_emb = nn.Embedding(num_embeddings=n_item, embedding_dim=dim)
        self.f = nn.Sigmoid()

    def get_embed(self, u):
        return self.U_emb(u)

    def get_rating(self, user):
        item = self.V_emb.weight.t()
        score = torch.matmul(user, item)
        return self.f(score)

    def forward(self, u):
        user = self.U_emb(u)
        item = self.V_emb.weight.t()
        score = torch.matmul(user, item)
        return self.f(score)


class MLP(nn.Module):
    def __init__(self, dim, layers):
        super(MLP, self).__init__()
        self.hidden = dim // 2
        self.layers = layers - 1
        self.net_in = nn.Sequential(
            nn.Linear(dim, self.hidden),
            nn.ReLU(),
        )
        self.net_hid = nn.Sequential(
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
        )
        self.net_out = nn.Sequential(
            nn.Linear(self.hidden, dim)
        )

    def forward(self, u):
        x = self.net_in(u)
        for _ in range(self.layers):
            x = self.net_hid(x)
        x = self.net_out(x)
        return x
