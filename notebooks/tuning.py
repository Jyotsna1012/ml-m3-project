import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib

# 1. Generate a synthetic dataset
X, y = make_classification(
    n_samples=1000,    # Number of samples
    n_features=20,     # Number of features
    n_informative=15,  # Number of informative features
    n_redundant=5,     # Number of redundant features
    random_state=42
)

# Convert to a pandas DataFrame for easier handling
X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(1, 21)])
y = pd.Series(y, name='target')

# 2. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Define the parameter grid for Random Forest
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# 4. Perform Grid Search for Hyperparameter Tuning
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)

# Fit the grid search to the training data
grid_search.fit(X_train, y_train)

# 5. Output the best parameters
best_params = grid_search.best_params_

# 6. Train the final model with the best parameters
best_rf = RandomForestClassifier(**best_params, random_state=42)
best_rf.fit(X_train, y_train)

# Save the best model
joblib.dump(best_rf, 'random_forest_model.pkl')

# 7. Evaluate the model on the test set
y_pred = best_rf.predict(X_test)

# Performance Metrics: Accuracy and F1-Score
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')  # Use 'weighted' for multi-class classification

# Display metrics
print(f"Test Set Accuracy: {accuracy:.4f}")
print(f"Test Set F1-Score: {f1:.4f}")

# 8. Generate the Report
report = f"""
# Hyperparameter Tuning Report

### Model: Random Forest Classifier

### 1. Hyperparameter Tuning Overview:
Hyperparameter tuning was performed to find the optimal parameters for the Random Forest Classifier model. The grid search method was used, evaluating multiple hyperparameter combinations to maximize the model's performance.

### 2. Parameter Grid:
The following hyperparameters were tuned:
- **n_estimators**: [50, 100, 150]
- **max_depth**: [10, 20, 30]
- **min_samples_split**: [2, 5, 10]
- **min_samples_leaf**: [1, 2, 4]

### 3. Best Parameters Found:
The best combination of hyperparameters found through grid search was:
- **n_estimators**: {best_params['n_estimators']}
- **max_depth**: {best_params['max_depth']}
- **min_samples_split**: {best_params['min_samples_split']}
- **min_samples_leaf**: {best_params['min_samples_leaf']}

### 4. Performance Metrics:
The model's performance on the test set using the best parameters:
- **Accuracy**: {accuracy:.4f}
- **F1-Score** (weighted): {f1:.4f}

### 5. Conclusion:
The hyperparameter tuning process successfully identified the optimal parameters for the Random Forest Classifier, leading to a final model with an accuracy of {accuracy:.4f} and an F1-score of {f1:.4f}. This model is ready for deployment.
"""

# Save the report to a markdown file
with open("hyperparameter_tuning_report.md", "w") as f:
    f.write(report)

print("Hyperparameter tuning report has been generated and saved as 'hyperparameter_tuning_report.md'")
