import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
import os
import warnings

from train import load_data, preprocess_data

warnings.filterwarnings('ignore')

def objective_lgbm(trial, X, y, cat_cols):
    params = {
        'objective': 'multiclass',
        'num_class': 4,
        'metric': 'multi_error',
        'random_state': 42,
        'class_weight': 'balanced',
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'num_leaves': trial.suggest_int('num_leaves', 10, 60),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'verbose': -1
    }
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in skf.split(X, y):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        clf = lgb.LGBMClassifier(**params)
        clf.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            categorical_feature=cat_cols,
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
        )
        
        preds = clf.predict(X_val)
        scores.append(accuracy_score(y_val, preds))
        
    return np.mean(scores)

def objective_catboost(trial, X, y, cat_cols):
    params = {
        'loss_function': 'MultiClass',
        'eval_metric': 'Accuracy',
        'random_seed': 42,
        'iterations': trial.suggest_int('iterations', 50, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'depth': trial.suggest_int('depth', 3, 7),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'verbose': False,
        'auto_class_weights': 'Balanced'
    }
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in skf.split(X, y):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        train_pool = Pool(X_train, y_train, cat_features=cat_cols)
        val_pool = Pool(X_val, y_val, cat_features=cat_cols)
        
        clf = CatBoostClassifier(**params)
        clf.fit(train_pool, eval_set=val_pool, early_stopping_rounds=20, verbose=False)
        
        preds = clf.predict(X_val)
        scores.append(accuracy_score(y_val, preds))
        
    return np.mean(scores)

if __name__ == "__main__":
    train_df, test_df = load_data()
    X, y, X_test, test_ids, features, cat_cols = preprocess_data(train_df, test_df)
    
    class_mapping = {'Fast': 0, 'Normal': 1, 'Slow': 2, 'Critical': 3}
    inv_class_mapping = {v: k for k, v in class_mapping.items()}
    y_encoded = y.map(class_mapping)
    
    print("--- Tuning LightGBM ---")
    study_lgb = optuna.create_study(direction='maximize')
    study_lgb.optimize(lambda trial: objective_lgbm(trial, X, y_encoded, cat_cols), n_trials=10)
    print(f"Best LGBM Accuracy: {study_lgb.best_value:.4f}")
    best_lgb_params = study_lgb.best_params
    best_lgb_params.update({'objective': 'multiclass', 'num_class': 4, 'class_weight': 'balanced', 'random_state': 42, 'verbose': -1})
    
    # Convert cat_cols to strings for CatBoost as it doesn't accept category dtype internally well sometimes, 
    # but the Pool with cat_features handles it if they are string/category.
    for col in cat_cols:
        X[col] = X[col].astype(str)
        X_test[col] = X_test[col].astype(str)
        
    print("\n--- Tuning CatBoost ---")
    study_cat = optuna.create_study(direction='maximize')
    study_cat.optimize(lambda trial: objective_catboost(trial, X, y_encoded, cat_cols), n_trials=10)
    print(f"Best CatBoost Accuracy: {study_cat.best_value:.4f}")
    best_cat_params = study_cat.best_params
    best_cat_params.update({'loss_function': 'MultiClass', 'random_seed': 42, 'verbose': False, 'auto_class_weights': 'Balanced'})
    
    # --- Final Training & Ensembling ---
    print("\n--- Training Final Ensemble ---")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    lgb_oof = np.zeros((len(X), 4))
    cat_oof = np.zeros((len(X), 4))
    test_preds_lgb = np.zeros((len(X_test), 4))
    test_preds_cat = np.zeros((len(X_test), 4))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded)):
        X_train_raw, y_train = X.iloc[train_idx], y_encoded.iloc[train_idx]
        X_val_raw, y_val = X.iloc[val_idx], y_encoded.iloc[val_idx]
        
        # Train LGB
        X_train_lgb = X_train_raw.copy()
        X_val_lgb = X_val_raw.copy()
        X_test_lgb = X_test.copy()
        for col in cat_cols:
            X_train_lgb[col] = X_train_lgb[col].astype('category')
            X_val_lgb[col] = X_val_lgb[col].astype('category')
            X_test_lgb[col] = X_test_lgb[col].astype('category')
            
        lgb_model = lgb.LGBMClassifier(**best_lgb_params)
        lgb_model.fit(X_train_lgb, y_train, eval_set=[(X_val_lgb, y_val)], categorical_feature=cat_cols, callbacks=[lgb.early_stopping(30, verbose=False)])
        lgb_oof[val_idx] = lgb_model.predict_proba(X_val_lgb)
        test_preds_lgb += lgb_model.predict_proba(X_test_lgb) / skf.n_splits
        
        # Train CatBoost
        X_train_cat = X_train_raw.copy()
        X_val_cat = X_val_raw.copy()
        X_test_cat = X_test.copy()
        for col in cat_cols:
            X_train_cat[col] = X_train_cat[col].astype(str)
            X_val_cat[col] = X_val_cat[col].astype(str)
            X_test_cat[col] = X_test_cat[col].astype(str)
            
        cat_model = CatBoostClassifier(**best_cat_params)
        cat_model.fit(Pool(X_train_cat, y_train, cat_features=cat_cols), eval_set=Pool(X_val_cat, y_val, cat_features=cat_cols), early_stopping_rounds=30)
        cat_oof[val_idx] = cat_model.predict_proba(X_val_cat)
        test_preds_cat += cat_model.predict_proba(X_test_cat) / skf.n_splits
        
    print(f"OOF Accuracy LGBM: {accuracy_score(y_encoded, np.argmax(lgb_oof, axis=1)):.4f}")
    print(f"OOF Accuracy CatBoost: {accuracy_score(y_encoded, np.argmax(cat_oof, axis=1)):.4f}")
    
    ensemble_oof = (lgb_oof + cat_oof) / 2.0
    print(f"OOF Accuracy Ensemble (Average): {accuracy_score(y_encoded, np.argmax(ensemble_oof, axis=1)):.4f}")
    
    # Save Submissions
    final_test_preds = np.argmax((test_preds_lgb + test_preds_cat) / 2.0, axis=1)
    submission = pd.DataFrame({'Request_ID': test_ids, 'Class': [inv_class_mapping[p] for p in final_test_preds]})
    os.makedirs('submissions', exist_ok=True)
    submission.to_csv('submissions/ensemble_submission.csv', index=False)
    print("Ensemble submission saved to submissions/ensemble_submission.csv")
