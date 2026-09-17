# Dataset

This project uses the ToN_IoT2020 dataset (or any CSV with the same shape:
numeric/categorical network-flow features + one label column).

Place your CSV here, e.g.:

    data/ton_iot.csv

Required:
- One column holding the multiclass attack label (e.g. `label`, `type`).
  Pass its name via `--label_col` if it isn't `label`.
- The rest of the columns are treated as features. Categorical columns
  (e.g. `proto`, `service`, `conn_state`) are automatically one-hot encoded.
- Identifier columns `ts`, `id`, `index` are dropped automatically if present.
