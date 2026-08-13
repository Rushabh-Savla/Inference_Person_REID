import torch, collections

p = "swin_base_msmt17.pth"
ck = torch.load(p, map_location="cpu", weights_only=False)
sd = ck.get("state_dict", ck.get("model", ck)) if isinstance(ck, dict) else ck

print("top-level keys:", list(ck.keys())[:10] if isinstance(ck, dict) else type(ck))
print("n tensors:", len(sd))
print("total params: %.1fM" % (sum(v.numel() for v in sd.values() if hasattr(v, "numel")) / 1e6))
for k, v in sd.items():
    if "classifier" in k or "bottleneck" in k or k.startswith("head"):
        print(k, tuple(v.shape))