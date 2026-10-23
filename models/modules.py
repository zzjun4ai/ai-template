import rootutils

root_path = rootutils.setup_root(__file__, indicator=".root", pythonpath=True)
from models.operations import *


class Dense(Module):
    def __init__(self, in_dim, out_dim, act="", norm="", dropout=0.0, bias=True, pre_norm=True):
        super().__init__()
        self.linear = Linear(in_dim, out_dim, bias=bias)
        self.act = str2act(act)
        self.norm = str2norm(norm, in_dim if pre_norm else out_dim)
        self.dropout = Dropout(dropout) if dropout > 0.0 else Identity()
        self.pre_norm = pre_norm

    def forward(self, x):
        if self.pre_norm:
            x = self.norm(x)
        x = self.linear(x)
        if not self.pre_norm:
            x = self.norm(x)
        return self.dropout(self.act(x))


class MLP(Module):
    def __init__(self, dims, act="", norm="", dropout=0.0, bias=True, pre_norm=False):
        super().__init__()
        n = len(dims)
        layers = [Dense(dims[i], dims[i + 1], act, norm, dropout, bias, pre_norm) for i in range(n - 2)]
        layers.append(Dense(dims[-2], dims[-1], bias=bias))
        self.layers = Sequential(*layers)

    def forward(self, x: Tensor):
        return self.layers(x)


class FFN(nn.Module):

    def __init__(self, in_dim, hidden_dim, act="silu", norm="layer", dropout=0.1):
        super().__init__()
        self.pre_norm = str2norm(norm, in_dim)
        self.layer1 = nn.Linear(in_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, in_dim)
        self.act = str2act(act)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.pre_norm(x)
        x = self.layer1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x


class SwiGLU(nn.Module):

    def __init__(self, in_dim, hidden_dim, act="silu", norm="layer", dropout=0.1):
        super().__init__()

        self.pre_norm = str2norm(norm, in_dim)
        self.in_proj = nn.Linear(in_dim, hidden_dim * 2)
        self.out_proj = nn.Linear(hidden_dim, in_dim)
        self.act = str2act(act)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.pre_norm(x)

        gate, value = self.in_proj(x).chunk(2, dim=-1)
        x = self.act(gate) * value

        x = self.dropout(x)
        x = self.out_proj(x)

        return x
