import joblib
import pandas as pd
import torch
from vajra.config import load_config
from vajra.models.registry import get_model
from vajra.flows.extractor import PcapFlowExtractor

# 1. Load Preprocessor and Model
pre = joblib.load("artifacts/preprocessor.joblib")
config = load_config("configs/model_transformer.yaml")
model = get_model(config, len(pre.feature_columns))
model.load_state_dict(torch.load("artifacts/checkpoints/baseline_model.pt", map_location="cpu"))
model.eval()

# 2. Extract flows from a REAL attack PCAP
# (You need a real PCAP containing an actual attack, like a DoS flood)
extractor = PcapFlowExtractor(feature_columns=pre.feature_columns, scaler=pre.scaler)
df_flows = extractor.extract_from_pcap("live_sample.pcap")

if df_flows.empty:
    print("No flows found.")
    exit()

# 3. Score the flows
x_scaled = extractor.transform_to_model_input(df_flows)
x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

with torch.no_grad():
    probs = torch.softmax(model(x_tensor), dim=1)[:, 1].numpy()

# 4. Report results
for i, (prob, flow) in enumerate(zip(probs, df_flows.itertuples())):
    verdict = "ATTACK" if prob > 0.880 else "BENIGN"
    print(f"Flow {i+1}: P(ATTACK)={prob:.4f} | Verdict={verdict} | Port={flow.Destination_Port}")