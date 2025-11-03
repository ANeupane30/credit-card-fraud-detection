from sklearn.preprocessing import RobustScaler
from load_data import df
from imblearn.over_sampling import SMOTE


"""  
Data Preprocessing

The dataset presents transcations that occurred in two days, where we have 492 frauds out of 284,807 transactions.
The dataset is highly unbalanced, with the positive class (frauds) accounting for 0.172% of all transactions.
The model tends to be biased towards the majority class (normal), which is non-fraudulent transactions. 
This imbalanced data, leading to poor predictive performance, especially for the minority class. 
Below are some strategies to address the issues of imbalanced data:

* Resampling Techniques: 
    a) Undersampling: Reducing the size of the majority class to match the minority class. This is done
    by randomly draw a subsample from the majority class. However, this may lead to loss of important information.
    
    b) Oversampling: Increasing the size of the minority class by duplicating instances or generating synthetic samples (e.g., using SMOTE).
    
* Class Weighting: Assigning higher weights to the minority class during model training to make it more
    influntial. This ensures that the model pays more attention to the minority class and adjusts its 
    decision boundary accordingly.

"""

# Feature Scaling 
"""
Given the prior PCA transformation of the other columns, it is imperative to scale the features Time and
Amount as well. RobustScaler is implemented for its ability to scale features using statistics that are 
robust to outliers. 
"""
# scale the columns 'Time' and 'Amount'
rs = RobustScaler()

df['scaled_amount'] = rs.fit_transform(df['Amount'].values.reshape(-1,1))
df['scaled_time'] = rs.fit_transform(df['Time'].values.reshape(-1,1))

df.drop(['Time', 'Amount'], axis=1, inplace=True)
print(df.head())
# print(df)


# Oversampling the minority class using SMOTE technique
X = df.drop('Class', axis=1)
y = df['Class']
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
df_resampled = X_resampled.copy()
df_resampled['Class'] = y_resampled
print("Original dataset shape:", df['Class'].value_counts())
print("Resampled dataset shape:", df_resampled['Class'].value_counts())
