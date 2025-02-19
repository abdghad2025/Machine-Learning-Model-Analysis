# Machine-Learning-Model-Analysis

This repository contains the implementation of multiple machine learning models—Ridge Regression, Lasso Regression, and Decision Tree—to predict fuel consumption based on various features. The code performs feature importance analysis, cross-validation, and error analysis to evaluate and interpret the performance of these models. Additionally, it provides insights into which features contribute the most to the prediction.

## Overview

This project aims to apply Ridge, Lasso, and Decision Tree regression models to predict fuel consumption based on input features such as speed, weather, and draft. It includes the following steps:

1. **Feature Importance Analysis**: Visualizes and interprets the most significant features for each model.
2. **Cross-Validation**: Evaluates the models using K-fold cross-validation and computes performance metrics such as Mean Absolute Error (MAE) and R² score.
3. **Error Analysis**: Analyzes residuals from the models to identify any patterns, biases, or outliers.
4. **Feature Importance Interpretation**: Summarizes the importance of each feature for all three models.

## Models Used

- **Ridge Regression**: A linear model with L2 regularization to prevent overfitting and enhance generalization.
- **Lasso Regression**: A linear model with L1 regularization to encourage sparsity in the model coefficients.
- **Decision Tree Regression**: A non-linear model that splits the data based on feature values to make predictions.

## Installation

To run this project, you will need Python 3.x and the following libraries:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

You can install the necessary dependencies using `pip`:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn

