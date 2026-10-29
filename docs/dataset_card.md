# Dataset Card

| Field | Value |
|---|---|
| Dataset | CIC-IDS2017 |
| Raw rows | 2,830,743 |
| Rows after cleaning | 2,520,798 |
| Rows dropped (NaN) | 2,867 |
| Rows dropped (duplicates) | 307,078 |
| Features (post-dedup) | 77 |
| Feature note | The raw CIC-IDS2017 CSVs contain a duplicate column `Fwd Header Length.1`. Dropping it yields 77 unique features, not the commonly cited 78. Keeping the duplicate would waste attention capacity and create an ambiguity for feature-attribution analysis. |
| Class balance | 83.1% benign, 16.9% attack |
| Split | 70/15/15 stratified, seed=42 |
| Preprocessor fit | Train split only (MinMaxScaler) |