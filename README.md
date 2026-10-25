# Credit Risk Lab

Credit Risk Lab is an end-to-end credit-risk classification project. It combines a reproducible data-preprocessing pipeline, three NumPy-based machine-learning implementations, model evaluation reports, exploratory data analysis, and a Django web dashboard for interactive predictions.

## Project purpose

The purpose of this project is to **compare the effectiveness of machine-learning models for credit-risk assessment**. It uses the same dataset, preprocessing workflow, and test set to evaluate K-Nearest Neighbors, Linear Regression, and Logistic Regression using Accuracy, Precision, Recall, F1-score, ROC/AUC, and confusion matrices. The Django dashboard presents the comparison results and supports risk predictions for new loan profiles.


## Features

- Cleans and validates a credit-risk dataset.
- Encodes categorical features using persisted mappings.
- Scales numerical features with min-max normalization.
- Splits data into stratified training and test sets.
- Implements K-Nearest Neighbors, Linear Regression, and Logistic Regression from scratch with NumPy.
- Evaluates models with Accuracy, Precision, Recall, F1-score, ROC, AUC, and confusion matrices.
- Generates exploratory data analysis charts and feature-importance visualizations.
- Provides a Django dashboard for:
  - entering a new loan profile;
  - comparing predictions from all three models;
  - viewing KNN-based risk probability and explanatory risk notes;
  - exploring dataset statistics and generated reports.
- Validates form input and restricts report-image access to known files.

## Project structure

```text
credit_risk_predictation/
├── data/
│   └── credit_risk_dataset.csv
├── preprocessing.py
├── model.py
├── runner.py
├── eda.py
├── params.json
├── requirements.txt
├── README.md
├── X_train.npy
├── X_test.npy
├── y_train.npy
├── y_test.npy
└── web_app/
    ├── manage.py
    ├── web_app/
    │   ├── settings.py
    │   ├── urls.py
    │   ├── asgi.py
    │   └── wsgi.py
    └── prediction/
        ├── admin.py
        ├── apps.py
        ├── ml_service.py
        ├── models.py
        ├── views.py
        ├── urls.py
        ├── tests.py
        ├── migrations/
        │   └── __init__.py
        ├── templates/
        │   └── index.html
        └── static/
            ├── images/
            │   └── generated report images
            └── prediction/
                ├── css/
                │   └── app.css
                └── js/
                    └── app.js
```


## Technology stack

- Python 3.11+ recommended
- Django 5.2+
- NumPy for numerical computation and model implementations
- Pandas for data loading and preprocessing
- scikit-learn for stratified train/test splitting
- Matplotlib and Seaborn for visualizations
- SQLite for the default Django database
- HTML, CSS, vanilla JavaScript, and Tailwind CSS via CDN for the dashboard

## Dataset and features

The source dataset is `data/credit_risk_dataset.csv`. The target column is `loan_status`:

- `0`: lower-risk or non-default profile
- `1`: high-risk or default profile

The model uses the following eleven features:

```text
person_age
person_income
person_home_ownership
person_emp_length
loan_intent
loan_grade
loan_amnt
loan_int_rate
loan_percent_income
cb_person_default_on_file
cb_person_cred_hist_length
```

## Machine-learning pipeline

### 1. Preprocessing

Run `preprocessing.py` to:

1. load the CSV dataset;
2. remove rows with missing values;
3. encode categorical columns;
4. min-max scale numerical columns;
5. create a stratified 80/20 train/test split;
6. save the processed arrays and preprocessing metadata.

Generated artifacts:

- `X_train.npy`
- `X_test.npy`
- `y_train.npy`
- `y_test.npy`
- `params.json`

`params.json` stores numerical min/max values, category mappings, feature order, and the target-column name. The web application uses this metadata to apply exactly the same transformations to new input.

### 2. Models

`model.py` contains three educational implementations:

- **K-Nearest Neighbors (`KNN`)**: predicts the majority label among the five closest training examples using Euclidean distance.
- **Linear Regression (`LinearRegression`)**: trained with gradient descent and converted to a binary classifier using a `0.5` threshold.
- **Logistic Regression (`LogisticRegression`)**: trained with gradient descent and a numerically guarded sigmoid function to produce risk probabilities.

### 3. Evaluation and reports

`runner.py` trains all models on the generated arrays and produces:

- per-model confusion matrices;
- a model-comparison chart;
- ROC curves and AUC values;
- Logistic Regression feature importance based on absolute coefficients.

`eda.py` produces:

- data-health summaries;
- age outlier analysis;
- class-distribution charts;
- numerical correlation heatmaps;
- feature-versus-target boxplots.

Generated images are saved in `web_app/prediction/static/images/`, which is the directory used by the Django dashboard.

## Web application

The Django application is located in `web_app/`. The central service in `prediction/ml_service.py`:

1. loads preprocessing metadata and NumPy arrays;
2. fits the three models once per application process;
3. validates and normalizes submitted form data;
4. returns predictions for all models;
5. calculates KNN risk probability from the five nearest training examples;
6. generates human-readable risk notes;
7. exposes dataset summaries and report metadata to the template.

The service is cached with `lru_cache`, so models are not retrained for every request.

## Installation

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

### Rebuild the preprocessing artifacts

```powershell
python preprocessing.py
```

### Train and evaluate the models

```powershell
python runner.py
```

### Run exploratory data analysis

```powershell
python eda.py
```

The EDA charts are also saved in `web_app/prediction/static/images/`. Both scripts resolve paths from their own location, so they can be run from the project root or another working directory.

### Start the Django dashboard

```powershell
cd web_app
python manage.py runserver
```

Open <http://127.0.0.1:8000/> in a browser.

## Tests

Run the Django test suite from the `web_app` directory:

```powershell
python manage.py test
```

The tests cover category mappings, default-profile prediction, report-path validation, page rendering, and form submission.

## Comparing model effectiveness

The quantitative results are printed by `runner.py`, and the comparison charts are saved in `web_app/prediction/static/images/`. Accuracy alone should not determine the best model for credit-risk assessment; Recall (the ability to detect defaults), Precision, F1-score, and ROC/AUC should be considered together.

