import struct
import math
import torch

# --- bit casting helpers ---
def float_to_u32(x: float) -> int:
    """Reinterpret float32 bits as uint32"""
    return struct.unpack('<I', struct.pack('<f', x))[0]

def u32_to_float(u: int) -> float:
    """Reinterpret uint32 bits as float32"""
    return struct.unpack('<f', struct.pack('<I', u & 0xFFFFFFFF))[0]


# --- FP32 -> FP24 (S1E8M15) truncation ---
def fp32_trunc_to_fp24(x: float) -> float:
    """
    FP32 -> FP24 (S1E8M15) by truncation:
    Keep top 24 bits, zero out low 8 bits.
    Return as float32 with low 8 bits cleared.
    """
    u = float_to_u32(x)

    if math.isnan(x):
        # canonical quiet NaN for FP32: 0x7FC00000
        u = 0x7FC00000
    else:
        u &= 0xFFFFFF00

    return u32_to_float(u)


# --- FP32 -> FP24 (S1E8M15) round-to-nearest-even ---
def fp32_rne_to_fp24(x: float) -> float:
    """
    FP32 -> FP24 (S1E8M15) using RNE rounding:
      lsb = (u >> 8) & 1
      bias = 0x7f + lsb
      u += bias
      u &= 0xffffff00
    Return as float32 with low 8 bits cleared.
    """
    u = float_to_u32(x)

    if math.isnan(x):
        u = 0x7FC00000
    else:
        lsb = (u >> 8) & 1
        rounding_bias = 0x7F + lsb
        u = (u + rounding_bias) & 0xFFFFFFFF
        u &= 0xFFFFFF00

    return u32_to_float(u)

# ------------------------------
# FP32 -> FP24 (S1E8M15) truncation
# Keep top 24 bits, clear low 8 bits
# ------------------------------
def fp32_trunc_to_fp24_torch(x: torch.Tensor) -> torch.Tensor:
    """
    x: float32 tensor
    returns: float32 tensor with low 8 bits cleared (FP24 emu by truncation)
    """
    if x is None:
        return x
    
    if x.dtype != torch.float32:
        x = x.to(torch.float32)

    # bitcast float->int32 (no copy)
    u = x.view(torch.int32)

    # mask to clear low 8 bits: 0xFFFFFF00 == -256 in int32
    mask = torch.tensor(-256, dtype=torch.int32, device=x.device)

    # canonical quiet NaN payload (fits in int32)
    qnan = torch.tensor(0x7FC00000, dtype=torch.int32, device=x.device)

    # clear low bits for non-NaN; set canonical qNaN for NaN
    u_out = torch.where(torch.isnan(x), qnan, u & mask)

    # bitcast back
    return u_out.view(torch.float32)


# ------------------------------
# FP32 -> FP24 (S1E8M15) RNE rounding
# lsb = (u >> 8) & 1
# bias = 0x7F + lsb
# u += bias; u &= 0xFFFFFF00
# ------------------------------
def fp32_rne_to_fp24_torch(x: torch.Tensor) -> torch.Tensor:
    """
    x: float32 tensor
    returns: float32 tensor with low 8 bits cleared (FP24 emu by RNE)
    """
    if x is None:
        return x
    
    if x.dtype != torch.float32:
        x = x.to(torch.float32)

    u = x.view(torch.int32)

    mask = torch.tensor(-256, dtype=torch.int32, device=x.device)        # 0xFFFFFF00
    qnan = torch.tensor(0x7FC00000, dtype=torch.int32, device=x.device)  # canonical qNaN
    bias_base = torch.tensor(0x7F, dtype=torch.int32, device=x.device)

    # NOTE: right shift on int32 is arithmetic; but for extracting bit 8,
    # it's fine because we immediately & 1.
    lsb = (u >> 8) & 1
    rounding_bias = bias_base + lsb

    u_rounded = (u + rounding_bias) & mask
    u_out = torch.where(torch.isnan(x), qnan, u_rounded)

    return u_out.view(torch.float32)

if __name__ == "__main__":
    vals = [1.0, 1.0000001, 0.1, -0.1, float('inf'), float('nan')]

    for v in vals:
        t = fp32_trunc_to_fp24(v)
        r = fp32_rne_to_fp24(v)
        print(v, "trunc:", t, "rne:", r)