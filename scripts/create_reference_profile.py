import torch
import joblib
import numpy as np
from vajra.config import load_config
from vajra.models.registry import get_model
from vajra.data.dataset import NIDSDataset
from vajra.justification.rollout import compute_attention_rollout, get_feature_attribution

def main():
    config = load_config("configs/model_transformer.yaml")
    pre = joblib.load("artifacts/preprocessor.joblib")
    
    # Load the model
    model = get_model(config, len(pre.feature_columns), store_attention=True)
    model.load_state_dict(torch.load("artifacts/checkpoints/baseline_model.pt", map_location="cpu"))
    model.eval()

    print("Loading 1000 benign training flows...")
    train_ds = NIDSDataset("data/processed/train.parquet", "artifacts/preprocessor.joblib")
    benign_df = train_ds.df[train_ds.df["binary_label"] == 0].head(1000)

    X = pre.transform(benign_df)
    tensor_x = torch.tensor(X, dtype=torch.float32)
    
    print("Running forward pass...")
    with torch.no_grad():
        # We add type: ignore here because Pylance sometimes thinks nn.Module is not callable
        model(tensor_x) 
        
    print("Extracting attention weights...")
    maps = model.get_attention_maps() # type: ignore
    rollout = compute_attention_rollout(maps)
    attn = get_feature_attribution(rollout).numpy()

    # Save the mean attention profile
    out_path = "artifacts/metrics/reference_profile.npy"
    np.save(out_path, np.mean(attn, axis=0))
    print(f"Success! Reference profile saved to {out_path}")

if __name__ == "__main__":
    main()