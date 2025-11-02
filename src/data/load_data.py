# Loading the datasets
import pandas as pd
from sklearn.ensemble import IsolationForest



def load_data(file_path):
    data = pd.read_csv(file_path)
    return data
df = load_data('../credit-card-fraud-detection/data/raw/creditcard.csv') 


# separating numerical and categorical columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# separating continuous and discrete numerical columns
continuous_cols = [col for col in num_cols if df[col].nunique() > 20]
discrete_cols = [col for col in num_cols if df[col].nunique() <= 20]


# Outlier Detection using Isolation Forest
iso_forest = IsolationForest(n_estimators=150, contamination='auto', random_state=42)
outliers = iso_forest.fit_predict(df[continuous_cols])
df['Outlier'] = outliers
print("Number of outliers detected:", sum(df['Outlier'] == -1))

# Calculating the percentage of outliers
outlier_percentage = (sum(df['Outlier'] == -1) / len(df)) * 100
print(f"Percentage of outliers in the dataset: {outlier_percentage:.2f}%")

# # Removing outliers
# df_cleaned = df[df['Outlier'] == 1].drop(columns=['Outlier'])
# print("Data shape after removing outliers:", df_cleaned.shape)


# saving the cleaned data
# check.to_csv('../credit-card-fraud-detection/data/processed/cleaned_data.csv', index=False)