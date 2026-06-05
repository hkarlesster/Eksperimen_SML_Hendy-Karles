import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import os

def run_data_pipeline(data_path, output_dir):
    print("🔄 Memulai pipeline data...")
    df = pd.read_csv(data_path)
    
    # 1. Handling Missing Values
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    cat_cols.remove('Heart Disease Status') # Target disendirikan
    
    imputer_num = SimpleImputer(strategy='median')
    df[num_cols] = imputer_num.fit_transform(df[num_cols])
    
    imputer_cat = SimpleImputer(strategy='most_frequent')
    df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])
    
    # 2. Encoding Categorical Features
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])
        
    # 3. Map Target
    df['Heart Disease Status'] = df['Heart Disease Status'].map({'Yes': 1, 'No': 0})
    
    # 4. Split Data
    X = df.drop(columns=['Heart Disease Status'])
    y = df['Heart Disease Status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 5. Scaling Numerical Features
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    
    # Simpan hasil pemrosesan
    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)
    print("✅ Pipeline data selesai! Berkas disimpan di:", output_dir)

if __name__ == "__main__":
    run_data_pipeline("dataset/heart_disease.csv", "dataset/processed")
