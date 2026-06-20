# Urgent Work Plan – Submission by Today 12:00 PM

**Goal**: Raise the leaderboard score from 0.7229 to **≥ 0.75** and generate a final `submission.csv` before noon.

---

## 1. Quick‑Win Actions (must be completed by 10:30 AM)
| Action | Reason | Approx. Time |
|--------|--------|--------------|
| **Add simple interaction features** (e.g., `year*month`, `dayofweek*is_weekend`) in `preprocess_data` | Gives models extra signal with almost no code | 15 min |
| **Threshold optimisation per class** – search best probability cut‑offs on the validation split | Often adds +0.005 – 0.008 accuracy | 20 min |
| **Stack LightGBM + CatBoost predictions** using out‑of‑fold (OOF) probabilities as meta‑features and train a lightweight Logistic Regression / MLP on top | Provides +0.010 – 0.015 accuracy | 45 min |
| **Generate final predictions** with the stacked model and write `submissions/final_submission.csv` | Final artefact for leaderboard | 10 min |

Total estimated effort: **≈ 1 hour 30 min**.

---

## 2. Detailed Steps
1. **Modify `train.py`**
   - Add the interaction columns in the `preprocess_data` function (lines ~30‑35).
   - After the existing LightGBM / CatBoost training loop, store the OOF probability matrices (`lgb_oof`, `cat_oof`).
   - Concatenate them (`stack_X = np.hstack([lgb_oof, cat_oof])`).
   - Train a small Logistic Regression (`sklearn.linear_model.LogisticRegression`) on `stack_X` using the same stratified‑CV splits to obtain meta‑model coefficients.
2. **Threshold optimisation**
   - Write a helper `optimize_thresholds(y_true, probs)` that uses a grid search (step 0.01) for each class to maximise overall accuracy on the validation set.
   - Apply the found thresholds to the stacked probabilities before `np.argmax`.
3. **Create final submission**
   - Use the trained meta‑model to predict probabilities for the test set (`test_stack = np.hstack([test_preds_lgb, test_preds_cat])`).
   - Apply the thresholds and write `Request_ID,Class` to `submissions/final_submission.csv`.
4. **Verification**
   - Print OOF accuracy of the stacked model and the threshold‑optimised accuracy.
   - Confirm the CSV shape (rows = test set, columns = 2) and that all IDs are present.

---

## 3. Timeline (today)
| Time | Milestone |
|------|-----------|
| **08:30 – 09:00** | Add interaction features in `train.py` and re‑run quick validation (should finish ~2 min). |
| **09:00 – 09:45** | Implement stacking logic, train Logistic Regression meta‑model, compute OOF score. |
| **09:45 – 10:15** | Write and run `optimize_thresholds` – obtain final accuracy (aim ≥ 0.75). |
| **10:15 – 10:30** | Generate `final_submission.csv` and double‑check format. |
| **10:30 – 11:00** | Quick sanity‑check on a few rows, back‑up all artefacts, commit changes. |
| **11:00 – 12:00** | Upload submission to the competition platform. |

---

## 4. Verification & Deliverables
- **Metric**: OOF accuracy of stacked + threshold‑optimised model should be **≥ 0.75**.
- **File**: `submissions/final_submission.csv` (two columns: `Request_ID,Class`).
- **Code artefacts**: Updated `train.py` (interaction features, stacking, thresholds) and a tiny helper script `threshold_opt.py` (optional). 
- **Documentation**: Add a short `README.md` block describing the quick‑win pipeline.

---

**If any step cannot be finished before the deadline, skip the optional threshold optimisation (the stacking alone already gives ~0.735) and submit the best model you have by 12:00 PM.**


**Goal**: Push the competition accuracy beyond the current 0.7229 (aim for ≥ 0.75) while respecting the 5‑submission limit.

---

## 1. Quick Wins (1‑2 submissions left)
| Idea | Rationale | Expected Gain |
|------|-----------|--------------|
| **Stacking / Blending** of LightGBM, CatBoost, and a simple neural net (FT‑Transformer) | Ensembles often capture complementary strengths. | +0.010 – 0.015 |
| **Threshold Optimization per Class** (instead of argmax) | Class‑wise probability thresholds can boost recall for minority classes. | +0.005 – 0.008 |
| **Pseudo‑Labeling** on the test set (high‑confidence predictions) | Expands training data without leakage. | +0.003 – 0.006 |
| **Feature‑wise Scaling & Interaction Features** (e.g., `year * month`, `dayofweek * is_weekend`) | Gives models richer signals. | +0.004 – 0.007 |

---

## 2. Medium‑Term Enhancements (If we still have submissions left)
1. **Advanced Text Embeddings**
   - Use pre‑trained Thai BERT (or `sentence‑transformers` multilingual) to obtain 768‑dim sentence vectors for the `เรื่องร้องทุกข์` field.
   - Reduce dimensionality with PCA/TruncatedSVD (≈ 100 dimensions) before feeding into tabular models.
2. **Tabular Neural Networks**
   - Implement **FT‑Transformer** or **TabNet** (GPU‑optional) – they excel on mixed categorical‑numeric‑text data.
3. **Extended Hyper‑Parameter Search**
   - Run Optuna with **200 trials** (instead of 10) for LightGBM and CatBoost, focusing on `feature_fraction`, `bagging_fraction`, and `min_child_samples`.
4. **Nested Stratified CV**
   - Use an outer 5‑fold CV for unbiased performance estimation and an inner 3‑fold CV for hyper‑parameter tuning. This prevents over‑fitting to the validation split.
5. **Class‑Balanced Loss**
   - Switch to **Focal Loss** for CatBoost and LightGBM (custom objective) to further mitigate class imbalance.

---

## 3. Long‑Term / High‑Risk Ideas (Only if we still have a submission slot)
| Idea | Why it may help | Risks |
|------|----------------|-------|
| **Synthetic Data Generation** (SMOTENC) for minority classes | Provides more samples for `Fast` and `Critical`. | May introduce noise; needs careful validation.
| **Feature Selection with SHAP/Permutation Importance** | Removes noisy features, possibly improving generalisation. | Extra compute; risk of discarding useful info.
| **Ensemble of Different Algorithms** (XGBoost, LightGBM, CatBoost, FT‑Transformer) with a meta‑learner (Logistic Regression) | Meta‑learner can learn optimal weighting. | Complexity; may exceed submission limit if many models.
| **Cross‑Validation Stacking** (K‑fold predictions as features) | Provides robust out‑of‑fold predictions for meta‑learning. | Heavy computation; high risk of leakage if not handled properly.

---

## 4. Revised Timeline (Assuming 2 submissions remaining)
| Day | Milestone |
|-----|-----------|
| **Day 1** | Implement stacking of existing LightGBM & CatBoost models + simple MLP; evaluate on hold‑out set. |
| **Day 2** | Tune class‑wise thresholds; generate submission #2. |
| **Day 3** | Add Thai BERT embeddings (batch‑process text, store vectors). |
| **Day 4** | Train FT‑Transformer on enriched feature set; perform 50‑trial Optuna search. |
| **Day 5** | Build final stacking/meta‑learner (using out‑of‑fold predictions from three models). |
| **Day 6** | Perform pseudo‑labeling on test set (top‑5% confident predictions), retrain, and generate final submission #3 (if still allowed). |
| **Day 7** | Validation, documentation, and backup of all artefacts. |

---

## 5. Validation & Verification
- **Metric**: Accuracy on a stratified 20 % hold‑out (same distribution as leaderboard). Aim for ≥ 0.75 before final submission.
- **Cross‑Check**: Ensure no leakage – `Request_ID` never appears in both train and validation splits.
- **Submission Checks**: Verify CSV format (`Request_ID,Class`) and that all IDs are present.
- **Reproducibility**: Seed all random generators (`random_state=42`). Save final model artefacts (`.pkl`) and a `requirements.txt` snapshot.

---

## 6. Deliverables
- Updated `train.py` with stacking logic and threshold optimisation.
- New `text_embeddings.py` to generate and cache BERT vectors.
- Revised `tune.py` with extended Optuna search.
- Final `submission.csv` placed in `submissions/` folder.
- Comprehensive `README.md` describing the pipeline, hyper‑parameters, and how to reproduce results.

---

**Prepared for the next iteration of the competition.**
