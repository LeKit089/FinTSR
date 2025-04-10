import torch
from mamba_ssm import Mamba

batch, length, dim = 18, 501, 512
x = torch.randn(batch, length, dim).to("cuda")
model = Mamba(
    # This module uses roughly 3 * expand * d_model^2 parameters
    d_model=dim, # Model dimension d_model
    d_state=16,  # SSM state expansion factor
    d_conv=4,    # Local convolution width
    expand=18,    # Block expansion factor
).to("cuda")
print(model)
y = model(x)
print(y.shape)
assert y.shape == x.shape