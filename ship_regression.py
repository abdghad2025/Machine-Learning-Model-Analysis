"""Ship-regression.ipynb

### Step 1.1: Load and Inspect Data

In this step, we will load the data from the CSV file and inspect its structure. We will:
1. Read the CSV file using pandas.
2. Display the first few rows of the dataset.
3. Provide descriptive statistics of the dataset.
4. Check for missing values.
"""

import pandas as pd

# Load and Inspect Data
def load_and_inspect_data(file_path):
    # Read the CSV file
    data = pd.read_csv(file_path)

    # Display the first few rows of the dataset
    print("First few rows of the dataset:")
    display(data.head())

    # Display descriptive statistics
    print("\nDescriptive statistics of the dataset:")
    display(data.describe())

    # Check for missing values
    print("\nMissing values in the dataset:")
    print(data.isnull().sum())

    return data

file_path = 'data/data.csv'
data = load_and_inspect_data(file_path)

"""### Step 1.2: Data Cleaning

In this step, we will clean the data by:
1. Handling missing values using forward fill.
2. Converting Unix time to a human-readable datetime format.

"""

# Data Cleaning
def clean_data(data):
    # Handle missing values (Example: Forward fill)
    data.ffill()

    # Convert Unix time to datetime
    data['Time'] = pd.to_datetime(data['Time'], unit='s')

    return data

data = clean_data(data)

# Display the cleaned data
print("Cleaned data:")
display(data.head())

"""### Step 1.3: Feature Engineering

In this step, we will engineer features to enhance the predictive power of our model:
1. Calculate the `Trim` feature as the difference between `Draft Forward (meters)` and `Draft Aft (meters)`.
2. Normalize/standardize the features to ensure they are on a comparable scale.

"""

from sklearn.preprocessing import StandardScaler

# Feature Engineering
def engineer_features(data):
    # Calculate Trim
    data['Trim'] = data['Draft Forward (meters)'] - data['Draft Aft (meters)']

    # Normalize/standardize features
    features_to_scale = [
        'Speed Over Ground (knots)', 'Speed Through Water (knots)',
        'Draft Forward (meters)', 'Draft Aft (meters)', 'Trim',
        'Weather Service True Wind Speed (knots)',
        'Weather Service Sea Current Speed (knots)'
    ]
    scaler = StandardScaler()
    data[features_to_scale] = scaler.fit_transform(data[features_to_scale])

    return data

data = engineer_features(data)

# Display the data with engineered features
print("Data with engineered features:")
display(data.head())

"""### Step 1.4: Feature Selection

In this step, we will select the relevant features for our model:
1. Select features based on domain knowledge and initial inspection.
2. Remove features that are not useful or highly correlated.

"""

# Feature Selection
def select_features(data):
    selected_features = [
        'Speed Over Ground (knots)', 'Speed Through Water (knots)',
        'Draft Forward (meters)', 'Draft Aft (meters)', 'Trim',
        'Weather Service True Wind Speed (knots)',
        'Weather Service Sea Current Speed (knots)'
    ]
    X = data[selected_features]
    y = data['Main Engine Fuel Consumption (MT/day)']

    return X, y

X, y = select_features(data)

# Display the selected features and target variable
print("Selected features:")
display(X.head())
print("Target variable:")
display(y.head())

"""Here's a step-by-step explanation of the output:

1. **Selected Features:**
   - The `select_features` function correctly extracts the specified features from the dataset.
   - The display of the selected features (`X.head()`) shows the first few rows of these features.

2. **Target Variable:**
   - The `y` variable is correctly set to the `Main Engine Fuel Consumption (MT/day)` column.
   - The display of the target variable (`y.head()`) shows the first few rows of this column.

Here is the correct and verified output:

### Selected Features:

| Speed Over Ground (knots) | Speed Through Water (knots) | Draft Forward (meters) | Draft Aft (meters) | Trim | Weather Service True Wind Speed (knots) | Weather Service Sea Current Speed (knots) |
|---------------------------|-----------------------------|------------------------|---------------------|------|------------------------------------------|--------------------------------------------|
| 0.940868                  | 0.977623                    | -3.048188              | -4.975085           | 6.019257 | -0.820678 | -0.458205 |
| 0.983655                  | 1.003407                    | -3.048188              | -4.975085           | 6.019257 | -0.764300 | -0.622453 |
| 1.013424                  | 0.992833                    | -3.048188              | -4.975085           | 6.019257 | -0.647687 | -0.570111 |
| 1.043192                  | 0.995337                    | -3.048188              | -4.975085           | 6.019257 | -0.533324 | -0.551036 |
| 1.049428                  | 0.990607                    | -3.048188              | -4.975085           | 6.019257 | -0.528069 | -0.563944 |

### Target Variable:

|  |
|--|
| 0.0 |
| 0.0 |
| 0.0 |
| 0.0 |
| 0.0 |

The data preparation steps have been executed correctly, and the selected features and target variable are ready for the next steps of model building and evaluation.

### Summary

We have successfully completed the data preparation steps which include loading and inspecting the data, cleaning the data, engineering new features, and selecting relevant features for our model. The next step will involve building and evaluating the machine learning model.

### Step 2.1: Train-Test Split

In this step, we will split the data into training and testing sets. This is important to evaluate the performance of our model on unseen data. We will:
1. Split the data into 80% training and 20% testing sets.
2. Use the `train_test_split` function from scikit-learn for this purpose.
"""

from sklearn.model_selection import train_test_split

# Train-Test Split
def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = split_data(X, y)

# Display the shapes of the training and testing sets
print(f"Training features shape: {X_train.shape}")
print(f"Testing features shape: {X_test.shape}")
print(f"Training target shape: {y_train.shape}")
print(f"Testing target shape: {y_test.shape}")

"""### Step 2.2: Model Selection and Training

In this step, we will select multiple models for prediction. We will:
1. Choose a simple linear regression model, Ridge, Lasso, and Decision Trees for comparison.
2. Train each model using the training dataset.

"""

from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.tree import DecisionTreeRegressor

# Model Selection and Training
def train_models(X_train, y_train):
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'Decision Tree': DecisionTreeRegressor()
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        models[name] = model

    return models

models = train_models(X_train, y_train)

# Display the models
for name, model in models.items():
    print(f"{name} model trained.")

"""### Step 2.3: Model Evaluation and Visualization

In this step, we will evaluate each model's performance using appropriate metrics. We will:
1. Predict the target values for the training and testing sets for each model.
2. Evaluate the models using metrics such as Mean Absolute Error (MAE) and R² score.
3. Compare the performance on the training and testing sets to check for overfitting.
4. Visualize the predictions versus the actual values to understand model performance.

"""

from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Model Evaluation and Visualization
def evaluate_models(models, X_train, X_test, y_train, y_test):
    results = {}

    for name, model in models.items():
        # Predict the target values for the training set
        y_train_pred = model.predict(X_train)
        # Predict the target values for the testing set
        y_test_pred = model.predict(X_test)

        # Evaluate the model
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        results[name] = {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'y_train_pred': y_train_pred,
            'y_test_pred': y_test_pred
        }

        print(f"{name} - Train MAE: {train_mae}, Test MAE: {test_mae}, Train R²: {train_r2}, Test R²: {test_r2}")

    return results

results = evaluate_models(models, X_train, X_test, y_train, y_test)

# Visualization
def plot_predictions(y_train, y_train_pred, y_test, y_test_pred, model_name):
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(y_train, y_train_pred, alpha=0.3)
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'{model_name} - Training Set: Actual vs Predicted')

    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_test_pred, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'{model_name} - Testing Set: Actual vs Predicted')

    plt.tight_layout()
    plt.show()

# Plot results for each model
for name, result in results.items():
    plot_predictions(y_train, result['y_train_pred'], y_test, result['y_test_pred'], name)

"""### Summary

We have successfully completed the model building steps which include:
1. Splitting the data into training and testing sets.
2. Selecting multiple models (Linear Regression, Ridge, Lasso, and Decision Trees) for comparison.
3. Training each model on the training dataset.
4. Evaluating each model using appropriate metrics such as Mean Absolute Error (MAE) and R² score.
5. Visualizing the predictions versus the actual values to understand model performance.

The visualizations help us compare the performance of different models and understand which model performs better on this dataset.

### Step 2.4: Model Evaluation

In this step, we will evaluate the model's performance using appropriate metrics. We will:
1. Predict the target values for the training and testing sets.
2. Evaluate the model using metrics such as Mean Absolute Error (MAE) and R² score.
3. Compare the performance on the training and testing sets to check for overfitting.
"""

from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Model Evaluation with Visualization
def evaluate_model(model, X_train, X_test, y_train, y_test):
    # Predict the target values for the training set
    y_train_pred = model.predict(X_train)
    # Predict the target values for the testing set
    y_test_pred = model.predict(X_test)

    # Evaluate the model
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print(f"Train MAE: {train_mae}")
    print(f"Test MAE: {test_mae}")
    print(f"Train R²: {train_r2}")
    print(f"Test R²: {test_r2}")

    # Visualize the predictions vs actual values
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(y_train, y_train_pred, alpha=0.3)
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Training Set: Actual vs Predicted')

    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_test_pred, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Testing Set: Actual vs Predicted')

    plt.tight_layout()
    plt.show()

    return train_mae, test_mae, train_r2, test_r2

train_mae, test_mae, train_r2, test_r2 = evaluate_model(model, X_train, X_test, y_train, y_test)

"""### Analysis of Model Performance and Visualization Results

In this section, we analyze the performance of various models: Linear Regression, Ridge, Lasso, and Decision Tree, based on their evaluation metrics and visualization of predictions versus actual values.

#### Linear Regression
- **Train MAE:** 9.38
- **Test MAE:** 9.28
- **Train R²:** 0.53
- **Test R²:** 0.55

The Linear Regression model shows moderate performance with both training and testing R² values around 0.53-0.55, indicating that the model explains about 53-55% of the variance in the data. The Mean Absolute Error (MAE) is around 9.28-9.38, which is reasonable but indicates room for improvement.

#### Ridge Regression
- **Train MAE:** 9.38
- **Test MAE:** 9.28
- **Train R²:** 0.53
- **Test R²:** 0.55

The Ridge Regression model performs almost identically to the Linear Regression model, as expected since it is a regularized version of the same. The R² and MAE values are nearly identical, showing similar explanatory power and prediction accuracy.

#### Lasso Regression
- **Train MAE:** 9.38
- **Test MAE:** 9.33
- **Train R²:** 0.52
- **Test R²:** 0.54

The Lasso Regression model also shows similar performance, with slightly lower R² values and slightly higher test MAE. This indicates that Lasso might be slightly underfitting the data compared to the Linear and Ridge models.

#### Decision Tree
- **Train MAE:** ~0
- **Test MAE:** 2.01
- **Train R²:** 1.0
- **Test R²:** 0.87

The Decision Tree model shows an excellent fit on the training data with an R² of 1.0 and near-zero MAE, indicating perfect prediction. However, on the testing data, it performs significantly better than the linear models with an R² of 0.87 and a much lower MAE of 2.01. This suggests that the Decision Tree model might be overfitting the training data but still provides the best predictive performance on the test set.

#### Visualizations
The scatter plots show actual vs. predicted values for each model on both the training and testing sets. The red dashed line represents the ideal scenario where predictions perfectly match the actual values.

- **Linear Regression and Ridge Regression**: The scatter plots for these models are very similar, with a moderate spread around the ideal line, indicating decent but not perfect predictions.
- **Lasso Regression**: Similar to Linear and Ridge but with slightly more spread in the test set predictions.
- **Decision Tree**: The training set predictions are perfect, but the test set predictions, while much closer to the ideal line, show some variability and potential overfitting.

Overall, while the Decision Tree model shows the best performance on the test set, the linear models provide more consistent and reliable predictions across both training and testing sets.

**Next Steps:**
1. Consider hyperparameter tuning for each model to improve performance further.
2. Explore ensemble methods like Random Forest or Gradient Boosting for potentially better predictive performance.
3. Perform feature importance analysis to understand which features contribute most to the model predictions.

### Summary

We have successfully completed the model building steps which include:
1. Splitting the data into training and testing sets.
2. Selecting a simple linear regression model for initial prediction.
3. Training the model on the training dataset.
4. Evaluating the model using appropriate metrics such as Mean Absolute Error (MAE) and R² score.

The next steps will involve further tuning of the model if necessary and visualizing the results.

### Step 3.1: Hyperparameter Tuning for Ridge, Lasso, and Decision Tree Models

In this step, we will perform hyperparameter tuning to improve the performance of our models. We will:
1. Use GridSearchCV from scikit-learn to find the best hyperparameters for Ridge, Lasso, and Decision Tree models.
2. Evaluate the performance of the tuned models.
"""

from sklearn.model_selection import GridSearchCV

# Hyperparameter Tuning for Ridge Regression
def tune_ridge(X_train, y_train):
    ridge = Ridge()
    parameters = {'alpha': [0.1, 1.0, 10.0, 100.0]}
    ridge_cv = GridSearchCV(ridge, parameters, cv=5, scoring='neg_mean_absolute_error')
    ridge_cv.fit(X_train, y_train)
    return ridge_cv.best_estimator_

# Hyperparameter Tuning for Lasso Regression
def tune_lasso(X_train, y_train):
    lasso = Lasso()
    parameters = {'alpha': [0.1, 1.0, 10.0, 100.0]}
    lasso_cv = GridSearchCV(lasso, parameters, cv=5, scoring='neg_mean_absolute_error')
    lasso_cv.fit(X_train, y_train)
    return lasso_cv.best_estimator_

# Hyperparameter Tuning for Decision Tree
def tune_decision_tree(X_train, y_train):
    tree = DecisionTreeRegressor()
    parameters = {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 10, 20],
        'min_samples_leaf': [1, 5, 10]
    }
    tree_cv = GridSearchCV(tree, parameters, cv=5, scoring='neg_mean_absolute_error')
    tree_cv.fit(X_train, y_train)
    return tree_cv.best_estimator_

# Tune the models
tuned_ridge = tune_ridge(X_train, y_train)
tuned_lasso = tune_lasso(X_train, y_train)
tuned_tree = tune_decision_tree(X_train, y_train)

# Display the best hyperparameters
print(f"Best Ridge parameters: {tuned_ridge.get_params()}")
print(f"Best Lasso parameters: {tuned_lasso.get_params()}")
print(f"Best Decision Tree parameters: {tuned_tree.get_params()}")

"""### Step 3.2: Evaluate Tuned Models

In this step, we will evaluate the performance of the tuned models using the training and testing datasets. We will:
1. Predict the target values for the training and testing sets using the tuned models.
2. Evaluate the models using metrics such as Mean Absolute Error (MAE) and R² score.
3. Visualize the predictions versus the actual values to understand model performance.

"""

# Evaluate Tuned Models
def evaluate_tuned_models(models, X_train, X_test, y_train, y_test):
    results = {}

    for name, model in models.items():
        # Predict the target values for the training set
        y_train_pred = model.predict(X_train)
        # Predict the target values for the testing set
        y_test_pred = model.predict(X_test)

        # Evaluate the model
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        results[name] = {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'y_train_pred': y_train_pred,
            'y_test_pred': y_test_pred
        }

        print(f"{name} - Train MAE: {train_mae}, Test MAE: {test_mae}, Train R²: {train_r2}, Test R²: {test_r2}")

    return results

# Models dictionary for evaluation
tuned_models = {
    'Ridge': tuned_ridge,
    'Lasso': tuned_lasso,
    'Decision Tree': tuned_tree
}

tuned_results = evaluate_tuned_models(tuned_models, X_train, X_test, y_train, y_test)

# Visualization
def plot_predictions(y_train, y_train_pred, y_test, y_test_pred, model_name):
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(y_train, y_train_pred, alpha=0.3)
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'{model_name} - Training Set: Actual vs Predicted')

    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_test_pred, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'{model_name} - Testing Set: Actual vs Predicted')

    plt.tight_layout()
    plt.show()

# Plot results for each tuned model
for name, result in tuned_results.items():
    plot_predictions(y_train, result['y_train_pred'], y_test, result['y_test_pred'], name)

"""### Analysis of Tuned Model Performance and Visualization Results

In this section, we analyze the performance of the tuned Ridge, Lasso, and Decision Tree models based on their evaluation metrics and visualization of predictions versus actual values.

#### Best Hyperparameters:
- **Ridge:**
  - alpha: 100.0
- **Lasso:**
  - alpha: 0.1
- **Decision Tree:**
  - max_depth: 30

#### Ridge Regression
- **Train MAE:** 9.38
- **Test MAE:** 9.27
- **Train R²:** 0.53
- **Test R²:** 0.55

The Ridge Regression model, after tuning, shows a slight improvement in performance. The R² values are around 0.53-0.55, indicating that the model explains about 53-55% of the variance in the data. The Mean Absolute Error (MAE) is around 9.27-9.38, which is consistent with the previous results but with a slight improvement.

#### Lasso Regression
- **Train MAE:** 9.37
- **Test MAE:** 9.28
- **Train R²:** 0.53
- **Test R²:** 0.55

The Lasso Regression model also shows a slight improvement after tuning. The R² values and MAE are similar to the Ridge model, indicating consistent performance across both models.

#### Decision Tree
- **Train MAE:** ~0
- **Test MAE:** 1.97
- **Train R²:** 1.0
- **Test R²:** 0.87

The tuned Decision Tree model shows an excellent fit on the training data with an R² of 1.0 and near-zero MAE, indicating perfect prediction on the training set. On the testing data, it performs significantly better than the linear models with an R² of 0.87 and a much lower MAE of 1.97. This suggests that the Decision Tree model provides the best predictive performance on the test set but might still be overfitting the training data.

#### Visualizations
The scatter plots show actual vs. predicted values for each model on both the training and testing sets. The red dashed line represents the ideal scenario where predictions perfectly match the actual values.

- **Ridge Regression**: The scatter plots for Ridge show a moderate spread around the ideal line, indicating decent but not perfect predictions.
- **Lasso Regression**: Similar to Ridge, with a moderate spread and consistent performance.
- **Decision Tree**: The training set predictions are perfect, but the test set predictions show some variability, indicating potential overfitting.

Overall, the Decision Tree model shows the best performance on the test set, while Ridge and Lasso provide consistent and reliable predictions across both training and testing sets.

**Next Steps:**
1. Consider further exploration of ensemble methods like Random Forest or Gradient Boosting for potentially better predictive performance.
2. Perform feature importance analysis to understand which features contribute most to the model predictions.

### Summary

We have successfully performed hyperparameter tuning for Ridge, Lasso, and Decision Tree models using GridSearchCV. The tuned models were then evaluated using the training and testing datasets. The evaluation metrics and visualizations show the improvements in model performance.

Next steps may include further exploration of ensemble methods or fine-tuning other models to achieve better predictive accuracy.

### Step 4.1: Feature Importance Analysis for Ridge and Lasso Models

In this step, we will analyze the feature importance for the Ridge and Lasso models. For linear models, the feature importance can be derived from the model coefficients.
"""

import pandas as pd
import numpy as np

# Extract feature importances from linear models
def get_linear_model_importance(model, feature_names):
    importance = model.coef_
    return pd.DataFrame({'Feature': feature_names, 'Importance': importance})

# Feature names
feature_names = X_train.columns

# Get feature importances for Ridge and Lasso
ridge_importance = get_linear_model_importance(tuned_ridge, feature_names)
lasso_importance = get_linear_model_importance(tuned_lasso, feature_names)

# Display feature importances
print("Ridge Regression Feature Importance:")
print(ridge_importance.sort_values(by='Importance', ascending=False))

print("\nLasso Regression Feature Importance:")
print(lasso_importance.sort_values(by='Importance', ascending=False))

"""### Step 4.2: Feature Importance Analysis for Decision Tree Model

For the Decision Tree model, we can use the `feature_importances_` attribute to get the importance of each feature.

"""

# Extract feature importances from the decision tree model
def get_tree_model_importance(model, feature_names):
    importance = model.feature_importances_
    return pd.DataFrame({'Feature': feature_names, 'Importance': importance})

# Get feature importances for Decision Tree
tree_importance = get_tree_model_importance(tuned_tree, feature_names)

# Display feature importances
print("Decision Tree Feature Importance:")
print(tree_importance.sort_values(by='Importance', ascending=False))

"""### Step 4.3: Visualize Feature Importance

In this step, we will visualize the feature importance for each model to better understand the impact of each feature on the prediction.

"""

import matplotlib.pyplot as plt

# Function to plot feature importances
def plot_feature_importance(importance_df, model_name):
    importance_df = importance_df.sort_values(by='Importance', ascending=True)
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.xlabel('Importance')
    plt.title(f'Feature Importance - {model_name}')
    plt.show()

# Plot feature importances for Ridge, Lasso, and Decision Tree
plot_feature_importance(ridge_importance, 'Ridge Regression')
plot_feature_importance(lasso_importance, 'Lasso Regression')
plot_feature_importance(tree_importance, 'Decision Tree')

"""### Summary of Feature Importance Analysis

We have successfully performed feature importance analysis for the Ridge, Lasso, and Decision Tree models. The feature importances help us understand which features contribute most to the model predictions.

#### Key Observations:

- **Ridge and Lasso Regression**:
  - Feature importances are derived from model coefficients.
  - The most important features are `Speed Over Ground (knots)` and `Speed Through Water (knots)`, indicating these speed measurements have the highest impact on fuel consumption predictions.
  - Other significant features include `Weather Service Sea Current Speed (knots)`, `Draft Aft (meters)`, and `Draft Forward (meters)`.

- **Decision Tree**:
  - Feature importances are derived from the `feature_importances_` attribute.
  - The most significant feature is `Speed Through Water (knots)`, which is the primary factor in making splits in the tree.
  - Other important features include `Draft Aft (meters)`, `Weather Service Sea Current Speed (knots)`, and `Draft Forward (meters)`.

#### Visualizations:

**Ridge Regression Feature Importance:**

- `Speed Over Ground (knots)` and `Speed Through Water (knots)` are the top features.
- `Weather Service Sea Current Speed (knots)` also has a notable impact.

**Lasso Regression Feature Importance:**

- Similar to Ridge, with `Speed Over Ground (knots)` being the most significant.
- `Speed Through Water (knots)` and `Draft Aft (meters)` follow.



**Decision Tree Feature Importance:**

- `Speed Through Water (knots)` is the most critical feature.
- Followed by `Draft Aft (meters)`, `Weather Service Sea Current Speed (knots)`, and `Draft Forward (meters)`.



#### Decision Tree Feature Importance Table:
| Feature                              | Importance |
|--------------------------------------|------------|
| Speed Through Water (knots)          | 0.645694   |
| Draft Aft (meters)                   | 0.116086   |
| Weather Service Sea Current Speed (knots) | 0.059123   |
| Draft Forward (meters)               | 0.059054   |
| Trim                                 | 0.053454   |
| Speed Over Ground (knots)            | 0.042174   |
| Weather Service True Wind Speed (knots) | 0.024415   |

Visualizations provide a clear understanding of feature importance, aiding in model interpretation and potential feature selection for future modeling efforts.

### Summary

We have successfully performed feature importance analysis for the Ridge, Lasso, and Decision Tree models. The feature importances help us understand which features contribute most to the model predictions.

Key observations:
- **Ridge and Lasso Regression**: Feature importances are derived from model coefficients. We can see which features have the highest positive or negative impact on the predictions.
- **Decision Tree**: Feature importances are derived from the `feature_importances_` attribute, showing the most significant features used in making splits in the tree.

Visualizations provide a clear understanding of feature importance, aiding in model interpretation and potential feature selection for future modeling efforts.

### Step 5.1: Cross-Validation

Cross-validation is a robust method to assess the generalizability of a model. It involves partitioning the data into subsets, training the model on some subsets while validating it on the remaining ones. This process is repeated several times, and the results are averaged to provide a more reliable estimate of model performance.

We will:
1. Use K-Fold Cross-Validation from scikit-learn.
2. Apply cross-validation to the Ridge, Lasso, and Decision Tree models.
3. Evaluate the models using Mean Absolute Error (MAE) and R² score.
"""

from sklearn.model_selection import cross_val_score

# Function to perform cross-validation
def cross_validate_model(model, X, y, cv=5):
    mae = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
    r2 = cross_val_score(model, X, y, cv=cv, scoring='r2')
    return -mae.mean(), r2.mean()

# Cross-validate Ridge, Lasso, and Decision Tree models
ridge_cv_mae, ridge_cv_r2 = cross_validate_model(tuned_ridge, X, y)
lasso_cv_mae, lasso_cv_r2 = cross_validate_model(tuned_lasso, X, y)
tree_cv_mae, tree_cv_r2 = cross_validate_model(tuned_tree, X, y)

# Display the cross-validation results
print(f"Ridge - Cross-Validated MAE: {ridge_cv_mae}, R²: {ridge_cv_r2}")
print(f"Lasso - Cross-Validated MAE: {lasso_cv_mae}, R²: {lasso_cv_r2}")
print(f"Decision Tree - Cross-Validated MAE: {tree_cv_mae}, R²: {tree_cv_r2}")

"""### Step 5.2: Error Analysis

Analyzing the errors in model predictions helps identify patterns or biases. By examining the residuals (the differences between actual and predicted values), we can gain insights into areas where the model performs well and where it may need improvement.

We will:
1. Calculate residuals for the training and testing sets.
2. Plot the residuals to identify any patterns or biases.
3. Perform a statistical analysis of the residuals to assess model performance.

"""

import seaborn as sns
import pandas as pd
import numpy as np

# Calculate residuals for the models
def calculate_residuals(model, X_train, X_test, y_train, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_residuals = y_train - y_train_pred
    test_residuals = y_test - y_test_pred
    return train_residuals, test_residuals

# Function to handle infinite values
def handle_infinite_values(data):
    return data.replace([np.inf, -np.inf], np.nan).dropna()

# Calculate residuals for Ridge, Lasso, and Decision Tree models
ridge_train_residuals, ridge_test_residuals = calculate_residuals(tuned_ridge, X_train, X_test, y_train, y_test)
lasso_train_residuals, lasso_test_residuals = calculate_residuals(tuned_lasso, X_train, X_test, y_train, y_test)
tree_train_residuals, tree_test_residuals = calculate_residuals(tuned_tree, X_train, X_test, y_train, y_test)

# Handle infinite values in residuals
ridge_train_residuals = handle_infinite_values(ridge_train_residuals)
ridge_test_residuals = handle_infinite_values(ridge_test_residuals)
lasso_train_residuals = handle_infinite_values(lasso_train_residuals)
lasso_test_residuals = handle_infinite_values(lasso_test_residuals)
tree_train_residuals = handle_infinite_values(tree_train_residuals)
tree_test_residuals = handle_infinite_values(tree_test_residuals)

# Plot residuals
def plot_residuals(train_residuals, test_residuals, model_name):
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    sns.histplot(train_residuals, kde=True)
    plt.title(f'{model_name} - Training Residuals')
    plt.xlabel('Residuals')

    plt.subplot(1, 2, 2)
    sns.histplot(test_residuals, kde=True)
    plt.title(f'{model_name} - Testing Residuals')
    plt.xlabel('Residuals')

    plt.tight_layout()
    plt.show()

# Plot residuals for each model
plot_residuals(ridge_train_residuals, ridge_test_residuals, 'Ridge Regression')
plot_residuals(lasso_train_residuals, lasso_test_residuals, 'Lasso Regression')
plot_residuals(tree_train_residuals, tree_test_residuals, 'Decision Tree')

"""### Step 5.3: Feature Importance

Understanding feature importance helps in interpreting the model and identifying the most influential features. For linear models like Ridge and Lasso, feature importance is derived from the model coefficients. For tree-based models, feature importance is derived from the frequency and significance of feature usage in decision nodes.

We will:
1. Assess feature importance using the coefficients for Ridge and Lasso models.
2. Assess feature importance using the `feature_importances_` attribute for the Decision Tree model.
3. Interpret the results to understand the impact of each feature on the predictions.

"""

# Feature importance has already been calculated and visualized earlier. Here, we summarize and interpret the results.

# Display feature importance for Ridge and Lasso models
print("Ridge Regression Feature Importance:")
print(ridge_importance.sort_values(by='Importance', ascending=False))

print("\nLasso Regression Feature Importance:")
print(lasso_importance.sort_values(by='Importance', ascending=False))

# Display feature importance for Decision Tree model
print("Decision Tree Feature Importance:")
print(tree_importance.sort_values(by='Importance', ascending=False))

"""### Detailed Analysis of Residuals and Cross-Validation Results

#### Residual Analysis

**Ridge Regression:**
- **Training Residuals:**
  - The histogram of residuals for the training set shows a peak around zero, indicating that the model is generally accurate but there are some large residuals (errors).
  - The distribution is slightly right-skewed, suggesting that there are more underestimations than overestimations.
  - There are several residuals that are significantly far from zero, indicating potential outliers or areas where the model performs poorly.
- **Testing Residuals:**
  - Similar to the training residuals, the test residuals also peak around zero with a right-skewed distribution.
  - The spread of residuals is narrower compared to the training set, suggesting that the model performs relatively better on unseen data.



**Lasso Regression:**
- **Training Residuals:**
  - The histogram shows a peak around zero, similar to Ridge Regression.
  - The distribution is right-skewed with some significant residuals far from zero.
  - The overall pattern is quite similar to Ridge, indicating that both models have similar performance characteristics on the training set.
- **Testing Residuals:**
  - The residuals for the test set also peak around zero and are right-skewed.
  - The spread is narrower than the training set, indicating better performance on unseen data.



**Decision Tree:**
- **Training Residuals:**
  - The histogram shows a sharp peak at zero, indicating perfect predictions on the training set.
  - This is a clear sign of overfitting, as the model has learned the training data too well.
- **Testing Residuals:**
  - The residuals for the test set show a wider spread with a peak around zero but many residuals significantly far from zero.
  - This indicates that the model's performance on unseen data is not as good, confirming overfitting.


#### Cross-Validation Results

- **Ridge Regression:**
  - **Cross-Validated MAE:** 10.54
  - **R²:** 0.47
  - The cross-validated MAE indicates moderate predictive performance with an R² of 0.47, suggesting that the model explains about 47% of the variance in the data.

- **Lasso Regression:**
  - **Cross-Validated MAE:** 10.53
  - **R²:** 0.47
  - Similar to Ridge Regression, Lasso shows moderate performance with nearly identical MAE and R² values.

- **Decision Tree:**
  - **Cross-Validated MAE:** 9.90
  - **R²:** 0.17
  - The Decision Tree model shows a lower MAE but a significantly lower R² of 0.17, indicating poor generalizability and overfitting.

#### Summary

- **Ridge and Lasso Regression:**
  - Both models perform similarly with moderate error rates and decent explanatory power.
  - Residuals analysis shows both models have a similar pattern of errors, with some outliers.
  - Cross-validation confirms that these models are relatively robust and reliable, with an R² of around 0.47.

- **Decision Tree:**
  - The model exhibits overfitting, with perfect performance on the training set but poor generalizability to the testing set.
  - Despite having a lower MAE in cross-validation, the low R² indicates that the model does not explain much variance in the data.

### Conclusion

Ridge and Lasso Regression models are preferable for this dataset due to their balance between error rates and generalizability. The Decision Tree model, while having a lower MAE, suffers from overfitting and poor explanatory power. Future steps should include exploring ensemble methods like Random Forest or Gradient Boosting to potentially improve performance and robustness.

### Future Steps for Comprehensive Modeling Task

Given an additional two weeks to spend on this modeling task, there are several strategies, improvements, and additional analyses that could be undertaken to enhance the robustness, accuracy, and interpretability of the model. The following steps outline a detailed plan for how to utilize this time effectively.

#### 1. **Data Enrichment and Feature Engineering:**
   - **Additional Data Sources:**
     - **Environmental Data:** Incorporate more detailed environmental data such as sea state (wave height, period, and direction), atmospheric pressure, and water temperature. These factors can significantly impact fuel consumption.
     - **Operational Data:** Collect data on ship operations such as engine load, maintenance records, and crew operations which might affect fuel efficiency.
     - **Port Data:** Information about port activities including docking times, loading/unloading durations, and port-specific conditions can provide insights into fuel consumption during port stays.

   - **Temporal Features:**
     - **Time of Day/Seasonal Effects:** Include features that capture time of day, day of the week, and seasonal variations which might influence operational strategies and environmental conditions.
     - **Lagged Features:** Generate lagged variables for key features to account for delayed effects. For instance, the speed and fuel consumption in previous hours can influence current fuel usage.

   - **Derived Features:**
     - **Interaction Terms:** Create interaction terms between features such as speed and draft to capture their combined effect on fuel consumption.
     - **Non-Linear Transformations:** Apply non-linear transformations to features (e.g., polynomial features) to capture more complex relationships.

#### 2. **Model Selection and Hyperparameter Tuning:**
   - **Advanced Models:**
     - **Ensemble Methods:** Implement Random Forest, Gradient Boosting Machines (GBM), and XGBoost to potentially improve predictive performance through ensemble learning.
     - **Neural Networks:** Explore deep learning models like neural networks which can capture complex non-linear relationships in the data.
   
   - **Hyperparameter Tuning:**
     - **Grid Search/Random Search:** Conduct an extensive hyperparameter tuning using Grid Search or Random Search to find the optimal settings for each model.
     - **Bayesian Optimization:** Utilize Bayesian optimization for more efficient hyperparameter tuning.

#### 3. **Model Evaluation and Validation:**
   - **Cross-Validation:**
     - **K-Fold Cross-Validation:** Perform K-Fold Cross-Validation with a larger number of folds to ensure robustness of model performance.
     - **Time Series Cross-Validation:** If data is sequential, use time series cross-validation techniques to better mimic real-world forecasting scenarios.
   
   - **Evaluation Metrics:**
     - **Additional Metrics:** Beyond MAE and R², consider using Root Mean Squared Error (RMSE), Mean Absolute Percentage Error (MAPE), and other relevant metrics.
     - **Error Analysis:** Conduct a detailed error analysis to identify specific conditions under which the model performs poorly.

#### 4. **Interpretability and Explainability:**
   - **Feature Importance:**
     - **SHAP Values:** Use SHapley Additive exPlanations (SHAP) values to interpret the impact of each feature on model predictions.
     - **LIME:** Apply Local Interpretable Model-agnostic Explanations (LIME) for local interpretation of model predictions.
   
   - **Visualization:**
     - **Partial Dependence Plots (PDP):** Create PDPs to visualize the effect of each feature on the predicted outcome.
     - **Feature Interaction Visualization:** Visualize interactions between key features to understand their combined impact on fuel consumption.

#### 5. **Model Deployment and Monitoring:**
   - **Deployment:**
     - **Pipeline Creation:** Develop a data preprocessing and model inference pipeline using tools like MLflow or Apache Airflow for reproducibility and scalability.
     - **API Development:** Implement the model as an API using Flask or FastAPI to allow for real-time predictions.
   
   - **Monitoring:**
     - **Performance Monitoring:** Set up a monitoring system to track model performance over time and detect any drift or degradation in performance.
     - **Retraining Strategy:** Establish a strategy for periodic retraining of the model with new data to maintain its accuracy and relevance.

#### 6. **Collaboration and Feedback:**
   - **Stakeholder Engagement:**
     - **Feedback Sessions:** Regularly engage with stakeholders (e.g., ship operators, engineers) to gather feedback on model predictions and its practical applicability.
     - **Domain Expertise:** Collaborate with domain experts to ensure that the model incorporates practical insights and addresses real-world challenges.

#### 7. **Documentation and Reporting:**
   - **Comprehensive Documentation:**
     - **Code Documentation:** Ensure that all code is well-documented with clear explanations of each step in the pipeline.
     - **Methodological Documentation:** Provide detailed documentation of the modeling process, including data preprocessing, feature engineering, model selection, and evaluation.
   
   - **Reporting:**
     - **Visual Reports:** Generate comprehensive reports with visualizations of model performance, feature importance, and error analysis.
     - **Executive Summaries:** Prepare executive summaries highlighting key findings, model benefits, and recommendations for future work.

### Additional Information or Data
1. **Real-Time Sensor Data:** Incorporating real-time sensor data from the ship could provide more granular insights into operational conditions.
2. **Historical Performance Data:** Historical performance data from similar voyages or different ships could be useful for comparative analysis.
3. **Fuel Quality Data:** Information about the quality of fuel used (e.g., sulfur content, energy density) could impact fuel consumption and should be considered.
4. **Economic Data:** Incorporating economic data such as fuel prices and shipping costs could provide a broader context for decision-making.

In conclusion, with an additional two weeks, a combination of enhanced data collection, advanced modeling techniques, thorough validation, and robust deployment strategies would significantly improve the predictive performance and practical utility of the fuel consumption model. This comprehensive approach ensures that the model is not only accurate but also interpretable, scalable, and aligned with real-world maritime operations.

### Future Steps for Comprehensive Modeling Task

Given an additional two weeks to spend on this modeling task, there are several strategies, improvements, and additional analyses that could be undertaken to enhance the robustness, accuracy, and interpretability of the model. The following steps outline a detailed plan for how to utilize this time effectively.

#### 1. **Data Enrichment and Feature Engineering:**
   - **Additional Data Sources:**
     - **Environmental Data:** Incorporate more detailed environmental data such as sea state (wave height, period, and direction), atmospheric pressure, and water temperature. These factors can significantly impact fuel consumption.
     - **Operational Data:** Collect data on ship operations such as engine load, maintenance records, and crew operations which might affect fuel efficiency.
     - **Port Data:** Information about port activities including docking times, loading/unloading durations, and port-specific conditions can provide insights into fuel consumption during port stays.

   - **Temporal Features:**
     - **Time of Day/Seasonal Effects:** Include features that capture time of day, day of the week, and seasonal variations which might influence operational strategies and environmental conditions.
     - **Lagged Features:** Generate lagged variables for key features to account for delayed effects. For instance, the speed and fuel consumption in previous hours can influence current fuel usage.

   - **Derived Features:**
     - **Interaction Terms:** Create interaction terms between features such as speed and draft to capture their combined effect on fuel consumption.
     - **Non-Linear Transformations:** Apply non-linear transformations to features (e.g., polynomial features) to capture more complex relationships.

#### 2. **Model Selection and Hyperparameter Tuning:**
   - **Advanced Models:**
     - **Ensemble Methods:** Implement Random Forest, Gradient Boosting Machines (GBM), and XGBoost to potentially improve predictive performance through ensemble learning.
     - **Neural Networks:** Explore deep learning models like neural networks which can capture complex non-linear relationships in the data.
   
   - **Hyperparameter Tuning:**
     - **Grid Search/Random Search:** Conduct an extensive hyperparameter tuning using Grid Search or Random Search to find the optimal settings for each model.
     - **Bayesian Optimization:** Utilize Bayesian optimization for more efficient hyperparameter tuning.

#### 3. **Model Evaluation and Validation:**
   - **Cross-Validation:**
     - **K-Fold Cross-Validation:** Perform K-Fold Cross-Validation with a larger number of folds to ensure robustness of model performance.
     - **Time Series Cross-Validation:** If data is sequential, use time series cross-validation techniques to better mimic real-world forecasting scenarios.
   
   - **Evaluation Metrics:**
     - **Additional Metrics:** Beyond MAE and R², consider using Root Mean Squared Error (RMSE), Mean Absolute Percentage Error (MAPE), and other relevant metrics.
     - **Error Analysis:** Conduct a detailed error analysis to identify specific conditions under which the model performs poorly.

#### 4. **Interpretability and Explainability:**
   - **Feature Importance:**
     - **SHAP Values:** Use SHapley Additive exPlanations (SHAP) values to interpret the impact of each feature on model predictions.
     - **LIME:** Apply Local Interpretable Model-agnostic Explanations (LIME) for local interpretation of model predictions.
   
   - **Visualization:**
     - **Partial Dependence Plots (PDP):** Create PDPs to visualize the effect of each feature on the predicted outcome.
     - **Feature Interaction Visualization:** Visualize interactions between key features to understand their combined impact on fuel consumption.

#### 5. **Model Deployment and Monitoring:**
   - **Deployment:**
     - **Pipeline Creation:** Develop a data preprocessing and model inference pipeline using tools like MLflow or Apache Airflow for reproducibility and scalability.
     - **API Development:** Implement the model as an API using Flask or FastAPI to allow for real-time predictions.
   
   - **Monitoring:**
     - **Performance Monitoring:** Set up a monitoring system to track model performance over time and detect any drift or degradation in performance.
     - **Retraining Strategy:** Establish a strategy for periodic retraining of the model with new data to maintain its accuracy and relevance.

#### 6. **Collaboration and Feedback:**
   - **Stakeholder Engagement:**
     - **Feedback Sessions:** Regularly engage with stakeholders (e.g., ship operators, engineers) to gather feedback on model predictions and its practical applicability.
     - **Domain Expertise:** Collaborate with domain experts to ensure that the model incorporates practical insights and addresses real-world challenges.

#### 7. **Documentation and Reporting:**
   - **Comprehensive Documentation:**
     - **Code Documentation:** Ensure that all code is well-documented with clear explanations of each step in the pipeline.
     - **Methodological Documentation:** Provide detailed documentation of the modeling process, including data preprocessing, feature engineering, model selection, and evaluation.
   
   - **Reporting:**
     - **Visual Reports:** Generate comprehensive reports with visualizations of model performance, feature importance, and error analysis.
     - **Executive Summaries:** Prepare executive summaries highlighting key findings, model benefits, and recommendations for future work.

### Additional Information or Data
1. **Real-Time Sensor Data:** Incorporating real-time sensor data from the ship could provide more granular insights into operational conditions.
2. **Historical Performance Data:** Historical performance data from similar voyages or different ships could be useful for comparative analysis.
3. **Fuel Quality Data:** Information about the quality of fuel used (e.g., sulfur content, energy density) could impact fuel consumption and should be considered.
4. **Economic Data:** Incorporating economic data such as fuel prices and shipping costs could provide a broader context for decision-making.

In conclusion, with an additional two weeks, a combination of enhanced data collection, advanced modeling techniques, thorough validation, and robust deployment strategies would significantly improve the predictive performance and practical utility of the fuel consumption model. This comprehensive approach ensures that the model is not only accurate but also interpretable, scalable, and aligned with real-world maritime operations.
"""