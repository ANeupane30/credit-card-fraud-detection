# Spliting data to train test split
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('../credit-card-fraud-detection/data/processed/preprocessed_data.csv')

# Defining the predictors and target variable
X = df.drop('Class', axis=1)
y = df['Class']

# spliting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set shape: {X_train.shape}, Testing set shape: {X_test.shape}")


from sklearn.ensemble import RandomForestClassifier
# from src.models.train import X_train, y_train, X_test

# # Random Forest model for prediction
# def predict(X_train, y_train, X_test):
#     model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
#     model.fit(X_train, y_train)
#     predictions = model.predict(X_test)
#     return predictions


# predict(X_train, y_train, X_test)