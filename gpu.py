import torch

# Check if CUDA is available and list GPUs
print("CUDA available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())
print("GPU Names:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])