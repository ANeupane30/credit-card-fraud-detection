# Spliting data to train test split
from sklearn.model_selection import train_test_split
from src.data.load_data import df

# Defining the predictors and target variable
X = df.drop('Class', axis=1)
y = df['Class']

# spliting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set shape: {X_train.shape}, Testing set shape: {X_test.shape}")
