
import torch
import numpy as np
from pathlib import Path
import sys

# Setup paths
project_root = Path("/home/mgarciade/projects/triage-mri")
sys.path.insert(0, str(project_root / "src"))

from triagemri.models.triage import build_triage_model
from triagemri.data.preprocessing import load_and_preprocess
import torch.nn.functional as F

def analyze_case(model, case_id, case_dir):
    t1c_path = list(case_dir.glob("*t1c.nii.gz"))[0]
    seg_path = list(case_dir.glob("*seg*.nii.gz"))[0]
    
    # Load and preprocess
    volume = load_and_preprocess(t1c_path).unsqueeze(0).cuda()
    
    # Inference
    model.eval()
    with torch.no_grad():
        output = model(volume)
        score = output["score"].item()
        attention = output["attention"].cpu().numpy().squeeze() # (27,)
    
    # Load GT
    import nibabel as nib
    seg_obj = nib.load(str(seg_path))
    seg = seg_obj.get_fdata()
    # Resize GT to 96x96x96 (matching input to MIL)
    seg_tensor = torch.from_numpy(seg).unsqueeze(0).unsqueeze(0).float()
    seg_96 = F.interpolate(seg_tensor, size=(96, 96, 96), mode="nearest").squeeze()
    
    # Total tumor volume in the 96x96x96 space
    total_tumor_voxels = float((seg_96 > 0).sum())
    
    # Split GT into 3x3x3 blocks
    # Each block is 32x32x32
    block_analysis = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                block = seg_96[i*32:(i+1)*32, j*32:(j+1)*32, k*32:(k+1)*32]
                tumor_voxels = float((block > 0).sum())
                tumor_pct = (tumor_voxels / total_tumor_voxels * 100) if total_tumor_voxels > 0 else 0
                block_analysis.append(tumor_pct)
    
    print(f"\nAnalysis for Case: {case_id}")
    print(f"Anomaly Score: {score:.4f}")
    
    # Numerical breakdown
    print("\nAttention vs Tumor Distribution (by 3x3x3 blocks):")
    print("Block (x,y,z) | Attention % | Tumor % in this block")
    print("-" * 50)
    
    sorted_idx = np.argsort(attention)[::-1]
    for idx in sorted_idx:
        attn_val = attention[idx]
        tumor_pct = block_analysis[idx]
        # Convert flat index to (x,y,z)
        z = idx // 9
        y = (idx % 9) // 3
        x = idx % 3
        if attn_val > 0.005 or tumor_pct > 0.1:
            print(f"({x},{y},{z})       | {attn_val*100:6.2f}%    | {tumor_pct:6.2f}%")

def main():
    project_root = Path("/home/mgarciade/projects/triage-mri")
    checkpoint = project_root / "outputs/default/checkpoints/last.ckpt"
    triad_weights = project_root / "weights/triad_swinb_simmim.pth"
    
    from triagemri.models.encoder import TriadEncoder
    from triagemri.models.mil import build_mil_head
    from triagemri.models.triage import TriageModel
    
    ckpt = torch.load(checkpoint, map_location="cpu")
    encoder = TriadEncoder(checkpoint_path=str(triad_weights), freeze=True, embed_dim=768)
    mil_head = build_mil_head(input_dim=768, hidden_dim=512, attention_dim=128, pooling="gated_attention")
    model = TriageModel(encoder=encoder, mil_head=mil_head)
    state_dict = {k[len("model."):] if k.startswith("model.") else k: v for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(state_dict, strict=False)
    model.cuda()
    
    # Specific case requested
    case_id = "BraTS-GLI-00098-001"
    case_dir = Path("triage-mri/data/raw/brain/brats/BraTS-GLI/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-00098-001")
    
    analyze_case(model, case_id, case_dir)

if __name__ == "__main__":
    main()
