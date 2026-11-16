import torch

ckpt = torch.load("artifacts/checkpoints/baseline_model.pt", map_location="cpu")
sd = ckpt if not isinstance(ckpt, dict) or "state_dict" not in ckpt else ckpt["state_dict"]

for name, tensor in sd.items():
    print(f"{name:50s} {tuple(tensor.shape)}")