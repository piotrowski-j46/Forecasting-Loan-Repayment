# Forecasting Loan Repayment

This repository contains the solution for the Data Science Club (PJATK) recruitment project.

The primary objective was to build a robust predictive model for a Credit Scoring classification problem, specifically designed to handle highly noisy data and a lack of obvious target dependencies.

# Architecture & Feature Engineering

The biggest challenge in this dataset was the absence of strong linear correlations between the features and the target variable. To extract the hidden signal, I applied an aggressive and deliberate feature engineering strategy rather than relying solely on the algorithm's brute force.

### Key Techniques:

- Advanced Categorical Encoding:
  - Target Encoding (OOF - Out of Fold): Applied strictly within cross-validation folds to completely eliminate data leakage.
  - Frequency Encoding: Used to capture signal from low-frequency categories.
- Discretization & Binning:
  - Rounding continuous values (to ones and tens) and grouping them into bins. This approach successfully extracted a clean, non-linear signal where linear correlations were near zero.
- Hyperparameter Optimization (HPO): Tuned the model using Optuna, focusing on reducing feature redundancy and preventing overfitting.
# Final Model (Ensemble)

The solution is powered by XGBoost. Instead of relying on a single "lucky" model, I implemented a variance-reduction architecture:
- 15-Model Ensemble: The final predictions use Seed Averaging across 15 models (3 random seeds × 5 StratifiedKFold splits).
- This approach ensures extremely low error variance and high robustness against test set randomness (Private LB shake-up).

# Tech Stack
- Python (pandas, numpy, scikit-learn, matplotlib, seaborn, json)
- XGBoost
- Optuna
