# Disease Prediction from Medical Data

Machine learning project for the CodeAlpha internship (Task 4).
Predicts whether a breast tumor is malignant or benign from patient data.

## Dataset
Breast Cancer Wisconsin (Diagnostic), loaded from scikit-learn.
569 samples, 30 numeric features, no missing values.
Classes: 212 malignant (label 0), 357 benign (label 1).

## Approach
1. Explored the data and checked class balance and missing values.
2. Stratified 80/20 train/test split; StandardScaler fitted on the training set only.
3. Compared Logistic Regression, SVM, Random Forest and XGBoost.
4. Evaluated with 5-fold stratified cross-validation (scaler inside a Pipeline to avoid data leakage).
5. Focused on Recall for the malignant class, since missing a malignant case is the most costly error.

## Results (5-fold cross-validation)
| Model | Accuracy | Recall (malignant) |
|---|---|---|
| SVM | 0.977 | 0.958 |
| Logistic Regression | 0.974 | 0.944 |
| XGBoost | 0.963 | 0.944 |
| Random Forest | 0.954 | 0.934 |

Final model: SVM (RBF kernel). On the held-out test set: accuracy 0.98, malignant recall 0.98, ROC-AUC 0.995.

Note: the test set is small (114 samples) and the models perform very close to each other,
so differences between them should not be over-interpreted.

## How to run
```bash
pip install -r requirements.txt
jupyter notebook disease_prediction.ipynb
```

## Files
- `disease_prediction.ipynb`: full analysis
- `disease_model.joblib`: trained pipeline (scaler + SVM), saved with scikit-learn 1.6.1
- `requirements.txt`: dependencies
