import torch
import torch.nn.functional as F


def to_complex(x: torch.Tensor) -> torch.Tensor:
    return x if torch.is_complex(x) else x.to(torch.complex64)

def complex_to_ri(x: torch.Tensor) -> torch.Tensor:
    x = to_complex(x)
    return torch.cat([x.real, x.imag], dim=1)  # [B,2,H,W] if input [B,1,H,W]

def ri_to_complex(x: torch.Tensor) -> torch.Tensor:
    r, i = torch.chunk(x, 2, dim=1)
    return torch.complex(r, i)

def fft2c(x: torch.Tensor) -> torch.Tensor:
    x = to_complex(x)
    x = torch.fft.ifftshift(x, dim=(-2, -1))
    X = torch.fft.fft2(x, norm="ortho")
    X = torch.fft.fftshift(X, dim=(-2, -1))
    return X

def ifft2c(X: torch.Tensor) -> torch.Tensor:
    X = to_complex(X)
    X = torch.fft.ifftshift(X, dim=(-2, -1))
    x = torch.fft.ifft2(X, norm="ortho")
    x = torch.fft.fftshift(x, dim=(-2, -1))
    return x

def A_forward(x: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    # x: [B,1,H,W] complex ; S: [B,C,H,W] complex -> [B,C,H,W] complex
    x = to_complex(x)
    S = to_complex(S)
    return fft2c(x * S)

def A_adjoint(k: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    # k: [B,C,H,W] complex ; S: [B,C,H,W] complex -> [B,1,H,W] complex
    k = to_complex(k)
    S = to_complex(S)
    img = ifft2c(k)
    return torch.sum(img * torch.conj(S), dim=1, keepdim=True)

def ensure_mask(mask: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    Accept mask shapes: [H,W], [B,H,W], [B,1,H,W], [B,C,H,W]
    Return: [B,1,H,W] float (0/1) broadcastable to coils.
    """
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    elif mask.dim() == 3:
        mask = mask.unsqueeze(1)               # [B,1,H,W]
    elif mask.dim() == 4:
        pass
    else:
        raise ValueError(f"mask dims not supported: {mask.shape}")

    if mask.shape[1] != 1:
        mask = (mask[:, :1] > 0).float()

    if mask.shape[0] == 1 and ref.shape[0] > 1:
        mask = mask.expand(ref.shape[0], -1, -1, -1)

    return mask.float()

def apply_mask(k: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = ensure_mask(mask, k)
    return k * mask