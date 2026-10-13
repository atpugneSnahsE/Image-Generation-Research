import platform
import sys

import torch
import diffusers
import transformers

print("Python:", sys.version)
print("Platform:", platform.platform())
print("PyTorch:", torch.__version__)
print("Diffusers:", diffusers.__version__)
print("Transformers:", transformers.__version__)
print("CUDA available:", torch.cuda.is_available())
print("MPS available:", torch.backends.mps.is_available())
print("CPU threads:", torch.get_num_threads())

x = torch.randn(512, 512)
y = x @ x.T

print("CPU tensor test:", tuple(y.shape))
print("Environment verification passed.")
