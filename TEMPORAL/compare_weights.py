
import torch
from pathlib import Path
import sys

project_root = Path("/home/mgarciade/projects/triage-mri")
sys.path.insert(0, str(project_root / "src"))

def compare():
    checkpoint = project_root / "outputs/default/checkpoints/last.ckpt"
    triad_weights_path = project_root / "weights/triad_swinb_simmim.pth"
    
    ckpt = torch.load(checkpoint, map_location="cpu")
    triad = torch.load(triad_weights_path, map_location="cpu")
    
    # Extract state dicts
    sd_ckpt = ckpt["state_dict"]
    sd_triad = triad["state_dict"] if "state_dict" in triad else triad
    
    # Compare one specific layer: swin.patch_embed.proj.weight
    # In ckpt it might be model.encoder.swin.patch_embed.proj.weight
    key_ckpt = "model.encoder.swin.patch_embed.proj.weight"
    key_triad = "backbone.swinViT.patch_embed.proj.weight"
    
    if key_ckpt in sd_ckpt and key_triad in sd_triad:
        w_ckpt = sd_ckpt[key_ckpt]
        w_triad = sd_triad[key_triad]
        
        diff = (w_ckpt - w_triad).abs().mean().item()
        print(f"Mean absolute difference in patch_embed: {diff:.8f}")
        if diff < 1e-6:
            print("BACKBONE WEIGHTS MATCH TRIAD! (Loaded correctly)")
        else:
            print("BACKBONE WEIGHTS DO NOT MATCH! (Random/Different)")
    else:
        print("Keys not found for comparison")
        print("CKPT keys start with:", list(sd_ckpt.keys())[:3])
        print("TRIAD keys start with:", list(sd_triad.keys())[:3])

if __name__ == "__main__":
    compare()
