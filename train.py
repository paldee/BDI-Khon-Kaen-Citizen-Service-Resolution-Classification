import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

def load_data():
    train_path = 'dataset/KKM_Citizen_Service_Train_Class.csv'
    test_path = 'dataset/KKM_Citizen_Service_Test.csv'
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def preprocess_data(train_df, test_df):
    # Combine for consistent preprocessing
    df = pd.concat([train_df, test_df], sort=False).reset_index(drop=True)
    
    # 1. Date Features
    # 'วันที่รับเรื่อง_clean' format seems to be string. Let's parse it
    df['วันที่รับเรื่อง_clean'] = pd.to_datetime(df['วันที่รับเรื่อง_clean'], errors='coerce', dayfirst=True)
    df['year'] = df['วันที่รับเรื่อง_clean'].dt.year
    df['month'] = df['วันที่รับเรื่อง_clean'].dt.month
    df['day'] = df['วันที่รับเรื่อง_clean'].dt.day
    df['dayofweek'] = df['วันที่รับเรื่อง_clean'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Interaction features
    df['year_month'] = df['year'] * 12 + df['month']
    df['day_weekend'] = df['dayofweek'] * df['is_weekend']
    
    # Fill any missing dates with median
    for col in ['year', 'month', 'day', 'dayofweek', 'is_weekend']:
        df[col] = df[col].fillna(df[col].median())
    
    # 2. Categorical Features
    cat_cols = ['ส่วนงาน', 'ฝ่าย', 'ประเภทคำร้อง', 'เขต', 'ชุมชน', 'สถานะ']
    for col in cat_cols:
        df[col] = df[col].fillna('Unknown').astype('category')
    
    # 3. Text Features (TF-IDF on 'เรื่องร้องทุกข์')
    df['เรื่องร้องทุกข์'] = df['เรื่องร้องทุกข์'].fillna('')
    tfidf = TfidfVectorizer(max_features=50, ngram_range=(1, 2))
    tfidf_features = tfidf.fit_transform(df['เรื่องร้องทุกข์'])
    
    # Use SVD to reduce text features if needed, or just use as is
    svd = TruncatedSVD(n_components=10, random_state=42)
    text_svd = svd.fit_transform(tfidf_features)
    for i in range(10):
        df[f'text_svd_{i}'] = text_svd[:, i]
    
    # Split back
    train_processed = df[df['Class'].notnull()].copy()
    test_processed = df[df['Class'].isnull()].copy()
    
    # Drop leak features and unnecessary ID columns for training
    drop_cols = ['Request_ID', 'เลขคำร้อง', 'เรื่องร้องทุกข์', 'วันที่รับเรื่อง', 
                 'วันที่เสร็จ', 'วันที่รับเรื่อง_clean', 'วันที่เสร็จ_clean', 'Resolution_Days', 'Class']
    
    features = [c for c in train_processed.columns if c not in drop_cols]
    
    X = train_processed[features]
    y = train_processed['Class']
    X_test = test_processed[features]
    
    return X, y, X_test, test_processed['Request_ID'], features, cat_cols

def train_evaluate(X, y, features, cat_cols):
    # Mapping Class to integers
    class_mapping = {'Fast': 0, 'Normal': 1, 'Slow': 2, 'Critical': 3}
    y_encoded = y.map(class_mapping)
    inv_class_mapping = {v: k for k, v in class_mapping.items()}
    
    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros((len(X), 4))
    
    models = []
    accuracies = []
    
    print(f"Training LightGBM on {len(X)} samples with {len(features)} features...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded)):
        X_train, y_train = X.iloc[train_idx], y_encoded.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y_encoded.iloc[val_idx]
        
        # Calculate sample weights to handle imbalance
        # We can pass class_weight='balanced' to LightGBM directly
        
        clf = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=4,
            random_state=42,
            class_weight='balanced',
            n_estimators=150,
            learning_rate=0.05,
            verbose=-1
        )
        
        clf.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            categorical_feature=cat_cols,
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
        )
        
        val_preds = clf.predict_proba(X_val)
        oof_preds[val_idx] = val_preds
        
        y_pred_class = np.argmax(val_preds, axis=1)
        acc = accuracy_score(y_val, y_pred_class)
        accuracies.append(acc)
        print(f"Fold {fold+1} Accuracy: {acc:.4f}")
        models.append(clf)
        
    final_preds_class = np.argmax(oof_preds, axis=1)
    oof_acc = accuracy_score(y_encoded, final_preds_class)
    print(f"\n--- OOF CV Accuracy: {oof_acc:.4f} ---")
    print("\nClassification Report:")
    print(classification_report(y_encoded, final_preds_class, target_names=['Fast', 'Normal', 'Slow', 'Critical']))
    
    return models, inv_class_mapping

def predict_and_save(models, X_test, test_ids, inv_class_mapping):
    print("Predicting on Test Dataset...")
    test_preds = np.zeros((len(X_test), 4))
    
    for clf in models:
        test_preds += clf.predict_proba(X_test) / len(models)
        
    final_test_preds = np.argmax(test_preds, axis=1)
    
    submission = pd.DataFrame({
        'Request_ID': test_ids,
        'Class': [inv_class_mapping[p] for p in final_test_preds]
    })
    
    os.makedirs('submissions', exist_ok=True)
    sub_path = 'submissions/baseline_lgbm_submission.csv'
    submission.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")

if __name__ == "__main__":
    train_df, test_df = load_data()
    X, y, X_test, test_ids, features, cat_cols = preprocess_data(train_df, test_df)
    models, inv_class_mapping = train_evaluate(X, y, features, cat_cols)
    predict_and_save(models, X_test, test_ids, inv_class_mapping)
