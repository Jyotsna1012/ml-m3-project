
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
- **n_estimators**: 150
- **max_depth**: 10
- **min_samples_split**: 2
- **min_samples_leaf**: 1

### 4. Performance Metrics:
The model's performance on the test set using the best parameters:
- **Accuracy**: 0.9000
- **F1-Score** (weighted): 0.9001

### 5. Conclusion:
The hyperparameter tuning process successfully identified the optimal parameters for the Random Forest Classifier, leading to a final model with an accuracy of 0.9000 and an F1-score of 0.9001. This model is ready for deployment.
