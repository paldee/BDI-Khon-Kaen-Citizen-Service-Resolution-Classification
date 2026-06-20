import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression
import os
import warnings

from train import load_data, preprocess_data

warnings.filterwarnings('ignore')

if __name__ == "__main__":
    print("Loading data with interaction features ...")
    train_df, test_df = load_data()
    X, y, X_test, test_ids, features, cat_cols = preprocess_data(train_df, test_df)
    class_mapping = {'Fast': 0, 'Normal': 1, 'Slow': 2, 'Critical': 3}
    inv_class_mapping = {v: k for k, v in class_mapping.items()}
    y_enc = y.map(class_mapping)

    # Best tuned parameters for LightGBM and CatBoost
    best_lgb_params = {
        'n_estimators': 60, 
        'learning_rate': 0.1537405590146472, 
        'max_depth': 5, 
        'num_leaves': 11,
        'subsample': 0.726585605972784, 
        'colsample_bytree': 0.7508762682407677,
        'reg_alpha': 2.2818509880456426, 
        'reg_lambda': 0.04242463155813183,
        'objective': 'multiclass', 
        'num_class': 4, 
        'class_weight': 'balanced',
        'random_state': 42, 
        'verbose': -1
    }
    
    best_cat_params = {
        'iterations': 200, 
        'learning_rate': 0.11952098442999953, 
        'depth': 6,
        'l2_leaf_reg': 7.551637350444674, 
        'bagging_temperature': 0.8312135876348736,
        'loss_function': 'MultiClass', 
        'random_seed': 42, 
        'verbose': False,
        'auto_class_weights': 'Balanced'
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    lgb_oof = np.zeros((len(X), 4))
    cat_oof = np.zeros((len(X), 4))
    test_lgb = np.zeros((len(X_test), 4))
    test_cat = np.zeros((len(X_test), 4))

    print("\n--- Level-1 Training (LightGBM & CatBoost) ---")
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y_enc)):
        print(f"Fold {fold+1}/5")
        X_tr_raw, y_tr = X.iloc[tr_idx], y_enc.iloc[tr_idx]
        X_va_raw, y_va = X.iloc[val_idx], y_enc.iloc[val_idx]

        # 1. Train LightGBM
        X_tr_lgb = X_tr_raw.copy()
        X_va_lgb = X_va_raw.copy()
        X_test_lgb = X_test.copy()
        for col in cat_cols:
            X_tr_lgb[col] = X_tr_lgb[col].astype('category')
            X_va_lgb[col] = X_va_lgb[col].astype('category')
            X_test_lgb[col] = X_test_lgb[col].astype('category')
        
        lgb_mod = lgb.LGBMClassifier(**best_lgb_params)
        lgb_mod.fit(X_tr_lgb, y_tr, eval_set=[(X_va_lgb, y_va)], categorical_feature=cat_cols,
                    callbacks=[lgb.early_stopping(30, verbose=False)])
        lgb_oof[val_idx] = lgb_mod.predict_proba(X_va_lgb)
        test_lgb += lgb_mod.predict_proba(X_test_lgb) / skf.n_splits

        # 2. Train CatBoost
        X_tr_cat = X_tr_raw.copy()
        X_va_cat = X_va_raw.copy()
        X_test_cat = X_test.copy()
        for col in cat_cols:
            X_tr_cat[col] = X_tr_cat[col].astype(str)
            X_va_cat[col] = X_va_cat[col].astype(str)
            X_test_cat[col] = X_test_cat[col].astype(str)
            
        cat_mod = CatBoostClassifier(**best_cat_params)
        cat_mod.fit(Pool(X_tr_cat, y_tr, cat_features=cat_cols),
                    eval_set=Pool(X_va_cat, y_va, cat_features=cat_cols),
                    early_stopping_rounds=30, verbose=False)
        cat_oof[val_idx] = cat_mod.predict_proba(X_va_cat)
        test_cat += cat_mod.predict_proba(X_test_cat) / skf.n_splits

    print(f"\nOOF Accuracy LGBM: {accuracy_score(y_enc, np.argmax(lgb_oof, axis=1)):.4f}")
    print(f"OOF Accuracy CatBoost: {accuracy_score(y_enc, np.argmax(cat_oof, axis=1)):.4f}")

    # --- Level-2 Stacking (Meta-Model) ---
    print("\n--- Level-2 Training (Logistic Regression Stacking) ---")
    stack_X = np.hstack([lgb_oof, cat_oof])
    test_stack_X = np.hstack([test_lgb, test_cat])
    oof_stack = np.zeros((len(X), 4))
    test_stack_pred = np.zeros((len(X_test), 4))
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y_enc)):
        X_tr, y_tr = stack_X[tr_idx], y_enc.iloc[tr_idx]
        X_va, y_va = stack_X[val_idx], y_enc.iloc[val_idx]
        
        meta = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        meta.fit(X_tr, y_tr)
        oof_stack[val_idx] = meta.predict_proba(X_va)
        test_stack_pred += meta.predict_proba(test_stack_X) / skf.n_splits
        
    acc_stack = accuracy_score(y_enc, np.argmax(oof_stack, axis=1))
    print(f"OOF Accuracy Stacked Model: {acc_stack:.4f}")

    # Generate final submission
    print("\nGenerating final submission...")
    final_preds = np.argmax(test_stack_pred, axis=1)
    submission = pd.DataFrame({'Request_ID': test_ids, 'Class': [inv_class_mapping[p] for p in final_preds]})
    os.makedirs('submissions', exist_ok=True)
    
    sub_path = 'submissions/final_solution_submission.csv'
    submission.to_csv(sub_path, index=False)
    print(f"Final solution submission saved to {sub_path}")
