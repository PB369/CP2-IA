import torch

print("PyTorch:", torch.__version__)
print("CUDA disponível:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA:", torch.version.cuda)
    print(
        "VRAM total:",
        torch.cuda.get_device_properties(0).total_memory / 1024**3,
        "GB"
    )
else:
    print("CUDA não está disponível.")