# Student Outcome Model Bench

A multi-class classification project that compares five machine-learning classifiers on the UCI **Predict Students' Dropout and Academic Success** dataset. The project includes a reproducible training pipeline, saved model artifacts, evaluation results, and an interactive Streamlit application.

> **Live Streamlit app:** [https://2026aj05412-nsp1-ml-assignment-2.streamlit.app/](https://2026aj05412-nsp1-ml-assignment-2.streamlit.app/)  
> **GitHub repository:** [https://github.com/hemxnth03/NSP1-ML-Assignment-2](https://github.com/hemxnth03/NSP1-ML-Assignment-2)

## a. Problem statement

This project classifies a student's recorded academic outcome as **Dropout**, **Enrolled**, or **Graduate** using demographic, prior-education, financial, academic-performance, and macroeconomic features. It is a three-class classification task. Five supervised classification algorithms are trained and compared using Accuracy, multiclass AUC, macro Precision, macro Recall, macro F1-score, and Matthews Correlation Coefficient (MCC). The comparison also examines how well each model identifies the smaller Enrolled class.

The dataset contains first- and second-semester academic-performance variables. Therefore, the experiment uses information collected during students' studies; it is **not an admission-time-only prediction system**.

## b. Dataset description

| Item | Detail |
|---|---|
| Dataset name | Predict Students' Dropout and Academic Success |
| Source | [UCI Machine Learning Repository — dataset 697](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) |
| Dataset DOI | [10.24432/C5MC89](https://doi.org/10.24432/C5MC89) |
| Licence | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| Instances | 4,424 student records |
| Input features | 36 |
| Target column | `Target` |
| Target classes | `Dropout`, `Enrolled`, `Graduate` |
| Missing values | None in the supplied dataset |

### Target class distribution

| Target class | Records | Share |
|---|---:|---:|
| Graduate | 2,209 | 49.93% |
| Dropout | 1,421 | 32.12% |
| Enrolled | 794 | 17.95% |
| **Total** | **4,424** | **100.00%** |

The target is imbalanced, with Enrolled as the smallest class. Macro-averaged Precision, Recall, and F1 are used so that each class contributes equally to those metrics.

### Feature overview

The 36 input features cover demographics and student background, application and course information, previous qualifications, parents' qualifications and occupations, tuition-fee and scholarship status, first- and second-semester curricular-unit activity and grades, and macroeconomic indicators.

### Preprocessing and train/test split

1. The source CSV is read as a semicolon-separated file. Whitespace is stripped from the column names.
2. `LabelEncoder` encodes the target. The saved class order is `Dropout`, `Enrolled`, `Graduate`.
3. The data is split into **80% training and 20% testing** using stratification and `random_state=2607`. This produces 3,539 training rows and 885 held-out test rows.
4. The 36 input columns are grouped into 9 nominal categorical features, 8 binary features, and 19 numeric/ordinal features.
5. Logistic Regression, KNN, and SVM use a scikit-learn `Pipeline` and `ColumnTransformer`: nominal features are one-hot encoded with infrequent-category handling, binary features pass through unchanged, and numeric/ordinal features are standardised with `StandardScaler`.
6. Decision Tree uses the raw input columns; scaling is not required for tree splits. Its integer-coded nominal features can still be treated as ordered numeric values. Gaussian Naive Bayes also uses the raw columns. One-hot 0/1 indicators are not naturally represented by its Gaussian feature likelihood, so this baseline retains the original columns; however, integer-coded nominal values are then treated as numeric inputs. Both choices are limitations to consider when interpreting the results.
7. Hyperparameters are selected with `GridSearchCV` using five-fold `StratifiedKFold` cross-validation and macro F1 as the scoring metric. Search is performed on the training split only; the hold-out test split is not used for model selection.
8. The five fitted estimators, including preprocessing pipelines where applicable, are saved under `model/`. The held-out test set, including its actual `Target` labels, is saved as `test_data.csv` for app evaluation.

### Evaluation metrics

- **Accuracy:** proportion of test records classified correctly.
- **AUC:** multiclass ROC AUC calculated using one-vs-rest (OvR) and macro averaging.
- **Precision, Recall, and F1:** macro-averaged across the three target classes.
- **MCC:** Matthews Correlation Coefficient, a correlation-based measure that uses all confusion-matrix categories.

## c. GitHub repository

**Repository link:** [https://github.com/hemxnth03/NSP1-ML-Assignment-2](https://github.com/hemxnth03/NSP1-ML-Assignment-2)

The repository contains the source code, dependency list, saved classifiers, model metadata, labelled hold-out test data, and experiment results.

### Repository structure

```text
project-folder/
├── .streamlit/
│   └── config.toml            Streamlit theme and client settings
├── model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   ├── support_vector_machine.joblib
│   └── meta.joblib            Feature names, class names, model filenames, sklearn version
├── results/
│   ├── metrics.csv            Held-out test metrics
│   └── best_params.json       Selected grid-search parameters and CV macro-F1
├── app.py                     Streamlit application
├── train_models.py            Training, tuning, evaluation, and model export
├── metrics_utils.py           Shared evaluation metric functions
├── requirements.txt           Pinned Python dependencies
├── test_data.csv              885-row labelled hold-out set for the app
├── README.md                  Project description and results
└── .gitignore
```

The original `data.csv` is used by `train_models.py` for model training and is excluded from Git by `.gitignore`. To reproduce training, download the dataset from the UCI page and place `data.csv` next to `train_models.py`. The deployed Streamlit application does not need the original training CSV because it loads the saved models.

### Reproducing the experiment

The reported training run was executed inside the project virtual environment using **Python 3.15.1**, confirmed with `python --version`. The dependency versions used by the run are pinned in `requirements.txt`.

In the activated project environment, check the Python version and install the pinned packages:

```powershell
python --version
# Python 3.15.1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Place the semicolon-separated UCI `data.csv` file in the project root, then run:

```powershell
python train_models.py
```

To start the app locally:

```powershell
streamlit run app.py
```

## d. Models used

All five classifiers were evaluated on the same **885-row stratified hold-out test set** generated with `random_state=2607`. The table below is taken from `results/metrics.csv` and matches the Streamlit app's **All five models** tab when `test_data.csv` is uploaded.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | **0.7921** | 0.8958 | **0.7454** | **0.7144** | **0.7248** | **0.6560** |
| Decision Tree | 0.7537 | 0.8597 | 0.6986 | 0.6738 | 0.6822 | 0.5935 |
| KNN | 0.7243 | 0.8326 | 0.6912 | 0.6218 | 0.6343 | 0.5394 |
| Naive Bayes | 0.6972 | 0.8088 | 0.6398 | 0.6173 | 0.6241 | 0.4955 |
| Support Vector Machine | 0.7864 | **0.8982** | 0.7349 | 0.7054 | 0.7155 | 0.6468 |

Bold values identify the highest value in each metric column. Logistic Regression leads on five metrics; SVM has the highest AUC.

### Selected hyperparameters

The following hyperparameters were selected by five-fold cross-validation on the training split:

| ML Model | Selected hyperparameters | Cross-validation macro F1 |
|---|---|---:|
| Logistic Regression | `C=0.3` | 0.6985 |
| Decision Tree | `max_depth=7`, `min_samples_leaf=15` | 0.6784 |
| KNN | `n_neighbors=7`, `weights='distance'` | 0.6066 |
| Gaussian Naive Bayes | `var_smoothing=1e-9` | 0.5972 |
| Support Vector Machine | `C=8`, `gamma=0.01` (RBF kernel) | 0.7023 |

### Per-class recall on the hold-out test set

| ML Model | Dropout recall | Enrolled recall | Graduate recall |
|---|---:|---:|---:|
| Logistic Regression | **0.7817** | **0.4340** | 0.9276 |
| Decision Tree | 0.6901 | 0.4151 | 0.9163 |
| KNN | 0.6655 | 0.2767 | 0.9231 |
| Naive Bayes | 0.7007 | 0.3208 | 0.8303 |
| Support Vector Machine | 0.7606 | 0.4214 | **0.9344** |

The test-set supports are 284 Dropout, 159 Enrolled, and 442 Graduate records. Enrolled is the most difficult class for all five models.

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | Logistic Regression achieved the best overall results, with 79.21% accuracy, 0.7248 macro F1-score and 0.6560 MCC. It also recorded the highest macro precision and macro recall among the five models. It correctly identified 69 out of 159 Enrolled students, although this class remains relatively difficult to predict. Overall, it gave the strongest results on most of the evaluation metrics. |
| **Decision Tree** | Decision Tree achieved 75.37% accuracy and a macro F1-score of 0.6822. Its performance was lower than Logistic Regression and SVM but better than KNN and Naive Bayes on all six metrics. One advantage is that its decision-making process is easier to interpret. However, using integer-coded categorical features as raw numerical values may affect how the tree creates its splits. |
| **KNN** | KNN achieved 72.43% accuracy and a macro F1-score of 0.6343. Although categorical encoding and feature scaling were applied, its performance was comparatively lower. It correctly identified only 44 out of 159 Enrolled students, giving it the lowest recall for this class. This suggests that the distance-based approach was less effective for distinguishing between the three student outcomes. |
| **Naive Bayes** | Gaussian Naive Bayes recorded the lowest accuracy (69.72%) and MCC (0.4955) among the five models. Its macro F1-score was also the lowest at 0.6241. One possible reason is that the Gaussian distribution and feature-independence assumptions may not adequately represent this dataset. The use of raw integer-coded categorical features is another limitation. |
| **Support Vector Machine** | SVM achieved 78.64% accuracy and a macro F1-score of 0.7155. It recorded the highest AUC of 0.8982, indicating good class discrimination. Its performance was close to Logistic Regression, although it scored slightly lower on accuracy, precision, recall, F1-score and MCC. It also achieved the highest recall for the Graduate class at 0.9344. |
| **Overall winner for this dataset** | Logistic Regression is selected as the overall winner because it achieved the highest scores in five out of the six evaluation metrics. SVM performed very closely, particularly in AUC, where it achieved a slightly higher score. The difference between the two models is small, so neither model should be considered significantly superior based on this single test split. Logistic Regression is preferred because of its slightly better overall results and simpler model structure. |

## e. Streamlit application

**Live app:** [https://2026aj05412-nsp1-ml-assignment-2.streamlit.app/](https://2026aj05412-nsp1-ml-assignment-2.streamlit.app/)

The Streamlit app loads the five saved `.joblib` classifiers and evaluates a labelled CSV uploaded by the user. It does not train or refit a model on uploaded data.

The uploaded CSV must contain all 36 input feature columns and the `Target` column with the actual labels (`Dropout`, `Enrolled`, or `Graduate`). The included `test_data.csv` is the 885-row hold-out set used for the results above. Uploading that file reproduces the comparison shown in the app.

The application provides:

- CSV upload and validation of required columns and target labels.
- A classifier dropdown for selecting one of the five saved models.
- Accuracy, AUC, macro Precision, macro Recall, macro F1, and MCC.
- A confusion matrix and classification report for the selected model.
- A side-by-side comparison table and metric chart for all five models.
- Student-level predictions, class probabilities, a misclassified-only filter, and CSV export.

### Deploy on Streamlit Community Cloud

1. Push the project files to a GitHub repository. Include `app.py`, `metrics_utils.py`, `requirements.txt`, `test_data.csv`, the `model/` artifacts, `.streamlit/config.toml`, `results/`, and this README.
2. Open [Streamlit Community Cloud](https://streamlit.io/cloud) and create a new app from the GitHub repository.
3. Select the repository, branch, and `app.py` as the application entry point.
4. In **Advanced settings**, choose a Python version from the dropdown. The local training environment used Python 3.15.1. The pinned `numpy` release needs Python 3.12 or newer, so do not select an older version.
5. Deploy the app and test it by uploading `test_data.csv`.

## f. Reproducibility and limitations

- `RANDOM_STATE` is fixed at `2607` in `train_models.py`.
- All five models use the same training/hold-out split. Hyperparameter selection is restricted to the training split.
- Preprocessing is fitted within the model pipelines for Logistic Regression, KNN, and SVM. The app passes raw feature columns to the saved pipelines.
- Decision Tree and Gaussian Naive Bayes use raw columns. Integer-coded nominal features are therefore treated as numeric values by these two models.
- The reported metrics are for one stratified hold-out split of 885 records. Small differences between models, especially Logistic Regression and SVM, should not be read as proof that one will be superior on other samples.
- `requirements.txt` pins the package versions. Keep the deployed scikit-learn version aligned with the version recorded in `model/meta.joblib` (`1.9.1`), because scikit-learn does not guarantee model-persistence compatibility across versions.
- The original `data.csv` is not required by the deployed app. It is only needed when rerunning `train_models.py`.

## References

1. Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). *Predict Students' Dropout and Academic Success* [Dataset]. UCI Machine Learning Repository. [https://doi.org/10.24432/C5MC89](https://doi.org/10.24432/C5MC89).
2. UCI Machine Learning Repository. [Dataset page](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success).