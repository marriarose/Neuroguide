import torch
import numpy as np
import scipy

print("PyTorch:", torch.__version__)
print("NumPy:", np.__version__)
print("SciPy:", scipy.__version__)
print("GPU available:", torch.cuda.is_available())