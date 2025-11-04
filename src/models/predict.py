from sklearn.ensemble import RandomForestClassifier
from src.models.train import X_train, y_train, X_test
from sklearn.model_selection import  cross_val_score, StratifiedKFold, GridSearchCV

# Random Forest model for prediction
model_rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model_rf.fit(X_train, y_train)
predictions = model_rf.predict(X_test)


# Compute the main metrics

# evaluate the recall score by straightforward k-fold cross-validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
recall_scores = cross_val_score(model_rf, X_train, y_train, cv=kf, scoring='recall')
print(f"Recall scores for each fold: {recall_scores}")  
print(f"Average Recall score: {recall_scores.mean()}")

# calculating precision score using cross-validation
precision_scores = cross_val_score(model_rf, X_train, y_train, cv=kf, scoring='precision')
print(f"Precision scores for each fold: {precision_scores}")

# Calculating F1-score using cross-validation
f1_scores = cross_val_score(model_rf, X_train, y_train, cv=kf, scoring='f1')
print(f"F1 scores for each fold: {f1_scores}")

# Calculating accuracy
accuracy_scores = cross_val_score(model_rf, X_train, y_train, cv=kf, scoring='accuracy')
print(f"Accuracy scores for each fold: {accuracy_scores}")