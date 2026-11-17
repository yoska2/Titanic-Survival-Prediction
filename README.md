# Titanic Survival Prediction

**Supervised Machine Learning | Binary Classification | Python**

## Project Overview

This project uses Titanic passenger data to predict whether a passenger survived (`1`) or did not survive (`0`). It explores survival patterns, trains a **Logistic Regression** model, compares model settings, and evaluates predictions on a separate test dataset.

## Technologies and Libraries

| Tool | Role in the project |
| --- | --- |
| **Python / Jupyter Notebook** | Runs the analysis, experiments, and model training. |
| **Pandas** | Loads CSV files and organizes training results. |
| **Matplotlib** | Creates charts for data exploration. |
| **Seaborn** | Visualizes survival rates by gender. |
| **Scikit-learn** | Provides model training, scaling, parameter search, and evaluation. |

## Main Machine Learning Methods

| Method | Purpose |
| --- | --- |
| **Logistic Regression** | Predicts one of two survival classes. |
| **StandardScaler** | Tests whether feature scaling improves the model. |
| **Pipeline** | Connects optional scaling with model training. |
| **GridSearchCV** | Compares combinations of model settings. |
| **5-Fold Cross-Validation** | Evaluates each configuration across five data splits. |
| **F1 Score** | Measures classification performance using precision and recall. |

The parameter search compares **scaling on/off**, regularization strength (`C = 0.1` or `1.0`), and penalty (`L2` or no penalty), using the `lbfgs` solver. The selected configuration uses **`C = 0.1`, L2 regularization, and no feature scaling**.

## Project Workflow

1. Load the training and test datasets using Pandas.
2. Explore survival rates by gender and ticket class with charts.
3. Separate passenger features from the `Survived` target.
4. Search for a suitable Logistic Regression configuration using GridSearchCV and 5-fold cross-validation.
5. Train the final pipeline on the full training dataset.
6. Predict survival on the test dataset and calculate its F1 score.

## Repository Structure

```text
Titanic-Survival-Prediction/
├── src/
│   └── Titanic_Survival_Prediction.ipynb
├── data/
│   ├── titanic_train.csv
│   └── titanic_test.csv
├── docs/
│   ├── Project_Summary.pdf
│   └── Technical_Documentation.pdf
├── README.md
└── .gitignore
```

 
## Running the Notebook

1. Install Python and Jupyter Notebook, along with `pandas`, `matplotlib`, `seaborn`, and `scikit-learn`.
2. Add `titanic_train.csv` and `titanic_test.csv` to `data/`.
3. Open `src/Titanic_Survival_Prediction.ipynb`.
4. **Update the two CSV paths in the data-loading cell** to match your notebook's working directory. Update the CSV file paths in the notebook to match your local directory structure.
5. Run the notebook cells in order.

**Dataset requirement:** Both CSV files must include a `Survived` column. The current notebook does not include an explicit step to encode categorical columns or fill missing values, so the feature columns must be suitable for Scikit-learn's Logistic Regression pipeline.

## Evaluation Note

The saved notebook reports the following results:

- **Cross-Validation F1: 0.842** — positive class: `0` (did not survive).
- **Final Test F1: 0.701** — positive class: `1` (survived).

These scores use different positive classes and evaluation stages, so they should **not be compared directly**. The final predictions are displayed alongside the actual survival outcomes.

## Documentation

- [Project Summary](docs/Project_Summary.pdf)
- [Technical Documentation](docs/Technical_Documentation.pdf)

## Project Explanation Video

[Watch the project video on YouTube](https://www.youtube.com/watch?v=CmiDEfPRvA8).
