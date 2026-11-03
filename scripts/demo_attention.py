import joblib, numpy as np, pandas as pd, torch, matplotlib.pyplot as plt
from vajra.config import load_config
from vajra.models.registry import get_model
from vajra.attacks.fgsm import fgsm_attack
from vajra.justification.rollout import compute_attention_rollout, get_feature_attribution

def main():
    config = load_config("configs/model_transformer.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pre = joblib.load("artifacts/preprocessor.joblib")
    
    # We MUST pass store_attention=True here!
    model = get_model(config, len(pre.feature_columns), store_attention=True).to(device)
    model.load_state_dict(torch.load("artifacts/checkpoints/baseline_model.pt", map_location=device))
    model.eval()

    df = pd.read_parquet("data/processed/test.parquet")
    flow = df[df["binary_label"] == 1].iloc[[0]] 
    x_clean = torch.tensor(pre.transform(flow), dtype=torch.float32).to(device)
    y_attack = torch.tensor([1], dtype=torch.long).to(device)
    
    with torch.no_grad():
        model(x_clean)
    clean_rollout = compute_attention_rollout(model.get_attention_maps())
    clean_weights = get_feature_attribution(clean_rollout)[0].cpu().numpy()

    eps = 0.05
    result = fgsm_attack(model, x_clean, y_attack, epsilon=eps)
    x_adv = result.adversarial
    
    with torch.no_grad():
        model(x_adv)
    adv_rollout = compute_attention_rollout(model.get_attention_maps())
    adv_weights = get_feature_attribution(adv_rollout)[0].cpu().numpy()

    features = pre.feature_columns
    top_clean_idx = np.argsort(clean_weights)[-10:][::-1]
    top_adv_idx = np.argsort(adv_weights)[-10:][::-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"What is the NIDS looking at? (FGSM eps={eps})", fontsize=16)

    ax1.barh([features[i] for i in top_clean_idx], [clean_weights[i] for i in top_clean_idx], color="tab:blue")
    ax1.set_title("Clean Attack Flow\n(Prediction: ATTACK)", fontsize=14)
    ax1.invert_yaxis()

    ax2.barh([features[i] for i in top_adv_idx], [adv_weights[i] for i in top_adv_idx], color="tab:red")
    ax2.set_title("Adversarial Flow\n(Prediction: BENIGN)", fontsize=14)
    ax2.invert_yaxis()

    plt.tight_layout()
    out_path = "artifacts/figures/attention_shift.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved visualization to {out_path}")

if __name__ == "__main__":
    main()