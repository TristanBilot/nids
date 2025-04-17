import torch

if torch.cuda.is_available():
    device = "cuda"
    print(f"Cuda used, with {torch.cuda.device_count()} GPUs")
else:
    device = "cpu"
    print("CPU is used")

COMPUTE_DEVICE = torch.device(device)
