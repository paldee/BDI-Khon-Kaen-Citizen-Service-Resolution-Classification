import pandas as pd
import numpy as np

train_path = 'dataset/KKM_Citizen_Service_Train_Class.csv'
test_path = 'dataset/KKM_Citizen_Service_Test.csv'

# Load data
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print("--- Training Data Info ---")
print(train_df.info())

print("\n--- Training Data Head ---")
print(train_df.head())

print("\n--- Class Distribution ---")
if 'Class' in train_df.columns:
    print(train_df['Class'].value_counts(dropna=False))

print("\n--- Missing Values ---")
print(train_df.isnull().sum())
