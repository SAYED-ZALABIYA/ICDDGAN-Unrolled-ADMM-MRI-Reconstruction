import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.fft import A_forward, A_adjoint, complex_to_ri, ri_to_complex
from utils.embedding import timestep_embedding, FiLM

class ProxNet(nn.Module):
    def __init__(self, base=32, emb_dim=128):
        super().__init__()
        self.conv = nn.Conv2d(2, base, 3, padding=1)
        self.rb = FiLM(emb_dim, base)
        self.out = nn.Conv2d(base, 2, 3, padding=1)

    def forward(self, x, emb):
        h = F.silu(self.conv(x))
        h = self.rb(h, emb)
        return x + self.out(h)

class UnrolledADMM(nn.Module):
    def __init__(self, K=8):
        super().__init__()
        self.K = K
        self.prox = ProxNet()
        self.rho = nn.Parameter(torch.ones(K))
        self.eta = nn.Parameter(torch.ones(K))

    def forward(self, x0, y, mask, S, t):
        x, z, u = x0, x0.clone(), torch.zeros_like(x0)
        emb = timestep_embedding(t, 128)
        for k in range(self.K):
            ksp = A_forward(z - u, S)
            ksp = ksp * (1 - mask) + y * mask
            x = A_adjoint(ksp, S)
            z = ri_to_complex(
                self.prox(complex_to_ri(x + u), emb)
            )
            u = u + self.rho[k] * (x - z)
        return z
