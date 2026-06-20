# Detailed Work Plan for Achieving Top‑3 on the Leaderboard

**Goal**: Develop a machine‑learning model that classifies municipal complaint requests into the four latency classes (Fast, Normal, Slow, Critical) with the highest possible accuracy, aiming for a top‑3 position on the competition leaderboard.

---

## 1. Project Overview
- **Target variable**: `Class` (4‑category classification).
- **Metrics**: Accuracy (0‑1 scale).
- **Submission format**: CSV with columns `Request_ID` and `Class`.
- **Constraints**:
  - Maximum 5 submissions.
  - No leakage of test labels.
  - Open‑source libraries only.

---

## 2. Timeline (14‑day sprint)
| Day | Milestone |
|-----|-----------|
| 1‑2 | Data exploration, cleaning, baseline model
| 3‑4 | Feature engineering & handling class imbalance
| 5‑7 | Advanced models (Gradient Boosting, Tabular NN) + hyper‑parameter tuning
| 8‑9 | Model ensembling & stacking
|10‑11| Robust validation (nested CV, stratified splits)
|12‑13| Final training on full data, generation of submission files
|14   | Documentation, backup, and final submission

---

## 3. Data Exploration & Understanding
1. Load `Training Dataset` and inspect:
   - `Request_ID`, categorical fields (`ส่วนงาน`, `ฝ่าย`, `ประเภทคำร้อง`, `เขต`, `ชุมชน`, `สถานะ`), date field (`วันที่รับเรื่อง`), and text field (`เรื่องร้องทุกข์`).
   - Count of each class; compute class distribution to assess imbalance.
2. Visualise key statistics:
   - Histogram of `Resolution_Days` (if available).
   - Frequency of categories.
3. Identify missing values, outliers, and inconsistent formatting.
4. Create a **Data Dictionary** markdown for future reference.

---

## 4. Data Pre‑processing
### 4.1 Categorical Encoding
- Use **Target‑Encoding** for high‑cardinality categories (e.g., `เรื่องร้องทุกข์` after text vectorisation).
- Apply **Ordinal / One‑Hot** encoding for low‑cardinality fields.

### 4.2 Date Features
- Parse `วันที่รับเรื่อง` → extract year, month, day of week, week‑of‑year, and holiday flags.
- Compute elapsed days from a reference date if needed.

### 4.3 Text Features
- Clean Thai text: remove punctuation, normalize whitespace.
- Generate **TF‑IDF** vectors (character n‑grams 2‑4) and **fastText** embeddings (pre‑trained Thai vectors).
- Optionally create **BERT‑Thai** sentence embeddings (if resources allow).

### 4.4 Missing Value Handling
- Impute categorical missing as a new category `Missing`.
- Impute numerical missing with median.

### 4.5 Scaling
- Apply **StandardScaler** or **RobustScaler** to numeric features (e.g., derived date diffs).

---

## 5. Handling Class Imbalance
- Compute class weights and pass to algorithms supporting weighted loss.
- Experiment with **SMOTE‑type** oversampling for tabular data (SMOTENC for mixed categorical‑numeric).
- Evaluate **Focal Loss** for neural nets.

---

## 6. Baseline Models
| Model | Reason |
|-------|--------|
| Logistic Regression (multinomial) | Quick sanity check, interpretability |
| Random Forest | Handles mixed data types, robust to outliers |
| XGBoost / LightGBM / CatBoost | State‑of‑the‑art for tabular data |
| Simple Feed‑forward NN | Baseline for deep learning |

- Use **5‑fold stratified CV** for baseline performance.
- Record baseline accuracy and confusion matrix.

---

## 7. Advanced Modeling
### 7.1 Gradient Boosting (primary candidate)
- **LightGBM** with categorical feature support.
- Tune `num_leaves`, `max_depth`, `learning_rate`, `feature_fraction`, `bagging_fraction`.
- Use **Optuna** for Bayesian hyper‑parameter optimization (max 100 trials).

### 7.2 CatBoost (handles categorical natively)
- Tune `depth`, `learning_rate`, `l2_leaf_reg`.
- Compare with LightGBM on validation folds.

### 7.3 Tabular Neural Networks
- **TabNet** or **FT‑Transformer** (if GPU available).
- Input: concatenated numeric + embedding of categorical + text embeddings.
- Use **weighted cross‑entropy**.

### 7.4 Ensemble Strategies
- **Weighted averaging** of probabilities from top 3 models.
- **Stacking**: meta‑learner (Logistic Regression) on out‑of‑fold predictions.
- Evaluate ensemble on held‑out validation.

---

## 8. Validation Strategy
- **Stratified Group K‑Fold**: group by `Request_ID` (ensuring no leakage of similar requests).
- Nested CV for hyper‑parameter tuning (inner loop for search, outer loop for unbiased estimate).
- Track **accuracy**, **confusion matrix**, and per‑class recall to avoid over‑optimising majority class.

---

## 9. Submission Pipeline
1. Fit the final ensemble on the **entire training set** using the best hyper‑parameters.
2. Predict probabilities for the **Test Dataset** (no `Class`).
3. Convert probabilities to class labels using the **class‑wise threshold** that maximises validation accuracy (or simply `argmax`).
4. Create `submission.csv` with `Request_ID` from test set and predicted `Class`.
5. Validate CSV format (header, correct ordering). Generate a checksum for sanity.
6. Keep a **git‑tracked** folder `submissions/` with timestamped files for audit.

---

## 10. Resource Management
- Use **conda/venv** environment with pinned versions:
  ```
  python==3.11
  pandas, numpy, scikit‑learn, lightgbm, catboost, optuna, fasttext, thai‑bert‑sentence‑transformer
  ```
- GPU (if available) for deep models; otherwise limit to CPU‑friendly algorithms.
- Store intermediate artefacts (feature matrices, encoders) in `artifacts/` with versioning.

---

## 11. Risk Mitigation & Contingencies
| Risk | Mitigation |
|------|------------|
| Over‑fitting to validation splits | Use robust CV, early stopping, limit model depth |
| Class imbalance causing poor minority recall | Apply class weighting, monitor per‑class metrics |
| Text preprocessing errors (Thai language) | Validate tokenisation on a sample, use pre‑trained Thai embeddings |
| Submission format issues | Automated CSV validator script before each upload |
| Exceeding 5 submission limit | Keep a log of submission scores; only submit after final validation |

---

## 12. Documentation & Handoff
- All notebooks/script files will contain **clear markdown headers** describing each step.
- Export the final pipeline as a **Python package** `bdi_classifier/` with `train.py`, `predict.py`, and `utils.py`.
- Include a `README.md` with usage instructions, environment setup, and reproducibility steps.
- Store the **detailed work plan** (this file) in the repository root for future contributors.

---

## 13. Success Criteria
- Validation accuracy ≥ 0.95 (target based on preliminary analysis).
- Top‑3 leaderboard placement after final submission.
- Reproducible pipeline that can be rerun end‑to‑end with a single command.

---

*Prepared for hand‑off to downstream model engineers.*
