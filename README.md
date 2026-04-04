# CatBoost — shouldSplit

Small ML project: binary classifier that predicts **`shouldSplit`** — i.e. whether a listing should be **split** into several service micro-categories or not.

## Data

Source is a CSV with listing fields. For the **first pipeline step** we **drop** columns that you won't have before a separate detector / labeling step: ids, the `split` column used for train/test splits in the dataset, `targetDetectedMcIds`, `targetSplitMcIds`, `caseType`.

What goes into **`X`** at minimum:

- `sourceMcId`, `sourceMcTitle` — placement category (passed as **CatBoost categoricals**).
- `description` — raw text (as **`text_features`** if you use CatBoost text handling).
- Hand-crafted numeric features from `description` (see below).

Target: **`shouldSplit`**.

---

## Feature extraction (`description`)

Implemented in `utils/description_features_utils.py`, joined to the frame in the notebook.

| Feature | What it is |
|--------|------------|
| `desc_words_count` | Word count via naive `str.split()` (whitespace). |
| `desc_length` | Char length of the description. |
| `has_bull_markers` | True if any of `•`, `-`, `/` shows up (list-ish / separator vibes). |
| `has_slashes` | True if `/` or `//` is present. |
| `slash_counted` | Count of `/`. |
| `paragraph_counter` | Count of `\n` newlines. |
| `is_word_separately_included` | Substring **«отдельно»** in lowercased text (Russian “separately”). |
| `should_split_words_trigger_counter` | Sum of phrase hits for “split / separate services” wording: *делаем отдельно*, *можем отдельно*, *отдельную*, *самостоятельную*, *выполняем отдельно*, *при необходимости*, *берем отдельно*, *также отдельно* — all counted on `lower()`. |
| `turnkey_count` | Sum of hits for *комплекс*, *не выезжаю*, *под ключ* (turnkey / “won’t come for one-off jobs” style phrases). |

Rough idea: length + structure of the text + Russian phrases that signal “several services listed separately” vs “full package” language.

---

## Model

- **`CatBoostClassifier`** — pick `loss_function` / `eval_metric` as you like (e.g. Logloss + AUC on val).
- Pass **`cat_features`** for categoricals, **`text_features`** for `description` if you actually use text columns in the pool.
- Save / load: **`model.save_model("*.cbm")`**, then **`CatBoostClassifier().load_model("*.cbm")`** — there is no `.save()`, it’s **`save_model`**.

---

## Metrics — how to read them

Everything is computed against labels **`y`** vs **`model.predict(X)`** or probs from **`model.predict_proba(X)`**.

- **Accuracy** — % of rows where predicted class == label. Easy number for a report; can lie if classes are imbalanced.
- **Precision (positive = “split”)** — of all “split” predictions, how many were really “split”. High precision ⇒ few **false splits** (we don’t flag normal one-service listings as “must split”).
- **Recall (positive = “split”)** — of all listings that **should** split, how many we catch. Low recall ⇒ **missed splits** (model says “don’t split” when it should).
- **F1** — harmonic mean of precision & recall on the positive class; single scalar when you need a balance.
- **ROC-AUC** — ranking quality from predicted **probabilities**; threshold-free, good for comparing models.
- **Log loss** — penalizes confident wrong answers; use **`predict_proba`**, not hard 0/1 labels.

**Confusion matrix** (sklearn default: rows = true class, cols = predicted, classes ordered e.g. `0` then `1`):

- Row **0** (no split): counts for predicted 0 / predicted 1.
- Row **1** (split): same.

**Missed split** = true **1**, predicted **0** (bottom-left cell if 0 = no split). **False split** = true **0**, predicted **1** (top-right).

---

## Splits

Use **train / val / test** for real projects: **val** for tuning + early stopping, **test** once at the end — data that never touched `fit` or the stopping logic.

---

## Repo layout

- `datasets/` — CSVs (large files: optional ignore in `.gitignore`).
- `notebooks/` — training notebook.
- `utils/` — text feature helpers.
