from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "data" / "credit_risk_dataset.csv"
REPORT_DIR = BASE_DIR / "web_app" / "prediction" / "static" / "images"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATASET_PATH)

def report_data_health():
    print("--- 1. DATASET OVERVIEW ---")
    print(df.info())
    
    print("\n--- 2. DESCRIPTIVE STATISTICS ---")
    print(df.describe())
    
    print("\n--- 3. MISSING VALUES ---")
    print(df.isnull().sum())

def perform_eda():
    print("\n--- GENERATING VISUALIZATIONS... ---")
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df['person_age'])
    plt.title('Age Distribution - Outlier Analysis')
    plt.savefig(REPORT_DIR / 'eda_age_outliers.png')
    
    plt.figure(figsize=(6, 4))
    sns.countplot(x='loan_status', data=df)
    plt.title('Class Distribution')
    plt.savefig(REPORT_DIR / 'eda_class_distribution.png')
    
    plt.figure(figsize=(12, 8))
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation Matrix')
    plt.savefig(REPORT_DIR / 'eda_correlation_matrix.png')
    
    features_to_plot = ['person_age', 'person_income', 'loan_amnt']
    for feat in features_to_plot:
        plt.figure(figsize=(8, 4))
        sns.boxplot(x='loan_status', y=feat, data=df)
        plt.title(f'Loan Status vs {feat}')
        plt.savefig(REPORT_DIR / f'eda_boxplot_{feat}.png')
        
    print("Complete. All visualizations have been saved to the project directory.")

if __name__ == "__main__":
    report_data_health()
    perform_eda()