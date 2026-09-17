# HeXAI-AttentionCPS

Runnable implementation of the methodology in Abdulganiyu et al., *"Explainable
attention based few shot LSTM for intrusion detection in imbalanced cyber
physical system networks"* (Scientific Reports, 2026), evaluated on ToN_IoT2020.

The authors' own GitHub repo (linked in the paper) exists but only contains
isolated notebook fragments — encoder, focal loss, attention layer, episode
sampler, and a partial episodic training/eval loop — with no data loader, no
full multiclass classifier, and no script wiring it all together. This
codebase keeps every piece of their logic **unchanged** (translated 1:1 from
notebook cells into modules) and adds what's missing to make it actually run
end to end. Additions are marked in each file's docstring:

- `preprocessing/load_data.py` — CSV loading, categorical encoding, split
- `models/fs_lstm.py::build_classifier` — full 10-class softmax head on top
  of the episodically-trained encoder (the original repo only evaluates via
  N-way K-shot episodes, which can't by itself produce the paper's
  per-class Table 6/7-style report)
- `training/train_classifier.py` — batch training for that head
- `evaluation/evaluate.py::evaluate_classifier` — full multiclass report
- `main.py` — end-to-end pipeline

## 1. Setup

```bash
cd hexai_attentioncps
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Tested against Python 3.9-3.11, TensorFlow 2.12-2.15.

## 2. Dataset

Put your CSV at `data/ton_iot.csv` (or anywhere — pass the path via `--csv`).
See `data/README.md`. You need:
- One label column with the attack-type string/int (e.g. `label`, `type`).
- Any mix of numeric + categorical flow features (categoricals like
  `proto`, `service`, `conn_state` are one-hot encoded automatically).

## 3. Run

Full pipeline (episodic pretraining → classifier head → evaluation → SHAP):

```bash
python main.py --csv data/ton_iot.csv --label_col label
```

Useful flags:

```bash
# Your label column is called something else
python main.py --csv data/ton_iot.csv --label_col type

# Skip SHAP while iterating (KernelExplainer is the slow part)
python main.py --csv data/ton_iot.csv --skip_shap

# Fewer episodes for a quick smoke test (paper: 3000 training / eval episodes)
# Edit config/config.yml's `episodes`, and pass:
python main.py --csv data/ton_iot.csv --fs_eval_episodes 20
```

Outputs:
- `results_multiclass.csv` — per-class Accuracy/Precision/Recall/F1/
  Specificity/FAR (Table 6/7 style)
- Console — episodic few-shot Accuracy/Precision/Recall/F1 (mean ± std)
- `shap_summary.png` — global SHAP feature-importance plot (Fig. 6 style),
  unless `--skip_shap`

## 4. Hyperparameters

`config/config.yml` mirrors the paper's Table 2 defaults (150 epochs, batch
64, lr 1e-3, focal alpha 0.25/gamma 1.9, PCA to 23 components, 4-way 5-shot
10-query episodes, 3000 episodes). The paper reports sweeping alpha/gamma
over a couple of value pairs — edit these directly to reproduce a sweep.

## 5. How the pieces fit together (matches Algorithm 1 / Fig. 1)

1. **Preprocessing**: min-max normalize → PCA to 23 components. Each row's
   23 PCA components are treated as a length-23 pseudo-sequence (shape
   `(23, 1)`) fed into the LSTM — this is what the original repo's
   `X[..., np.newaxis]` reshape does, and matches the paper's `input_shape`.
2. **Phase 1 — episodic few-shot encoder training**: `LSTM(128)->Dropout->
   LSTM(64)->AttentionLayer` produces an embedding per sample. Each episode
   samples N-way K-shot support/query sets **from minority classes only**,
   computes class prototypes from the support embeddings, classifies the
   query set by nearest prototype, and backpropagates class-wise focal loss.
3. **Phase 2 — classifier head**: a softmax `Dense(num_classes)` layer is
   trained on top of the pretrained encoder with focal loss over the full
   (batch, not episodic) training set, so every class — not just minority
   ones — gets a usable decision boundary.
4. **Evaluation**: both the paper's episodic few-shot metrics and a standard
   one-vs-rest multiclass report (Accuracy/Precision/Recall/F1/Specificity/
   FAR per class) are computed.
5. **Explainability**: SHAP `KernelExplainer` on the flattened classifier,
   matching Fig. 6's global summary plot.

## 6. Notes / known rough edges

- `evaluate_fewshot` runs `num_episodes` fresh forward passes with
  `model.predict`, which is slow for large `num_episodes` — the paper's 3000
  is mostly meant for *training*; use a smaller number (e.g. 200-500) for
  evaluation unless you have GPU time to spare.
- SHAP `KernelExplainer` scales roughly with `len(X_explain) * nsamples`;
  keep `--shap_samples` small (10-50) for iteration.
- The PCA-components-as-timesteps framing is a design choice inherited
  directly from the official repo's reshape; the paper's text is not fully
  explicit about how tabular flow records become an LSTM sequence.
