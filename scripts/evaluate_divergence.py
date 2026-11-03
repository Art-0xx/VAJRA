# Evaluate Attention Entropy as an adversarial detector.
import joblib
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.stats import entropy
from sklearn.metrics import roc_curve, auc

from vajra.config import load_config
from vajra.models.registry import get_model
from vajra.attacks.fgsm import fgsm_attack
from vajra.justification.rollout import compute_attention_rollout, get_feature_attribution

def get_entropy(model, x):
    with torch.no_grad():
        model(x)
    rollout = compute_attention_rollout(model.get_attention_maps())
    attn = get_feature_attribution(rollout) # Shape: (batch, 77)
    
    # Normalize to make it a proper probability distribution
    attn = attn / attn.sum(dim=-1, keepdim=True)
    
    # Compute Shannon Entropy for each flow in the batch
    ent = entropy(attn.cpu().numpy(), axis=-1)
    return ent

def main():
    config = load_config("configs/model_transformer.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pre = joblib.load("artifacts/preprocessor.joblib")
    
    model = get_model(config, len(pre.feature_columns), store_attention=True).to(device)
    model.load_state_dict(torch.load("artifacts/checkpoints/baseline_model.pt", map_location=device))
    model.eval()

    # 1. Load 1000 clean attack flows
    print("Loading test data...")
    test_df = pd.read_parquet("data/processed/test.parquet")
    test_attacks = test_df[test_df["binary_label"] == 1].head(1000)
    x_clean = torch.tensor(pre.transform(test_attacks), dtype=torch.float32).to(device)
    y_attack = torch.ones(len(x_clean), dtype=torch.long).to(device)

    # 2. Generate Adversarial Flows (FGSM eps=0.05)
    print("Generating adversarial flows...")
    result = fgsm_attack(model, x_clean, y_attack, epsilon=0.05)
    x_adv = result.adversarial

    # 3. Compute Entropies
    print("Computing Attention Entropy...")
    clean_ent = get_entropy(model, x_clean)
    adv_ent = get_entropy(model, x_adv)

    # 4. Evaluate Detector (Higher entropy = more likely adversarial)
    y_true = np.concatenate([np.zeros(len(clean_ent)), np.ones(len(adv_ent))])
    y_scores = np.concatenate([clean_ent, adv_ent])
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    print("\n=== ATTENTION ENTROPY DETECTOR ===")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Clean Entropy mean: {np.mean(clean_ent):.4f} (Should be low)")
    print(f"Adv Entropy mean:   {np.mean(adv_ent):.4f} (Should be high)")
    print(f"Optimal Threshold:  {optimal_threshold:.4f}")

    # 5. Plot
    plt.figure(figsize=(8, 5))
    plt.hist(clean_ent, bins=30, alpha=0.6, label="Clean Flows (Focused)", color="green")
    plt.hist(adv_ent, bins=30, alpha=0.6, label="Adversarial Flows (Confused)", color="red")
    plt.axvline(optimal_threshold, color="black", linestyle="--", label=f"Threshold ({optimal_threshold:.3f})")
    plt.title("Attention Entropy Distribution")
    plt.xlabel("Shannon Entropy (Higher = More Spread Out)")
    plt.ylabel("Number of Flows")
    plt.legend()
    plt.tight_layout()
    
    out_path = "artifacts/figures/attention_entropy.png"
    plt.savefig(out_path, dpi=200)
    print(f"\nSaved histogram to {out_path}")

if __name__ == "__main__":
    main()