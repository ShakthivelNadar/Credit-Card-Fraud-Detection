# Credit-Card-Fraud-Detection
Credit Card Fraud Detection is an artificial intelligence and machine learning application that identifies fraudulent transactions in real time by analyzing customer spending behavior and transaction patterns
Credit Card Fraud Detection
Project Overview
This project focuses on building a machine learning model to detect fraudulent credit card transactions. Due to the highly imbalanced nature of fraud detection datasets, special techniques like SMOTE (Synthetic Minority Over-sampling Technique) are employed to handle the class imbalance and improve model performance.

Dataset
The dataset used for this project is creditcard.csv, which contains anonymized credit card transactions. The features V1, V2, ..., V28 are the result of a PCA transformation, and Time and Amount are the only original features not transformed. The Class column is the target variable, where 0 represents legitimate transactions and 1 represents fraudulent transactions.

Data Characteristics:
Highly Imbalanced: The dataset exhibits a severe class imbalance, with a very small percentage of transactions being fraudulent.
Anonymized Features: Most features (V1-V28) are principal components obtained via PCA.
Features: Time, Amount, V1-V28, Class.
Preprocessing Steps
Feature Scaling: The Time and Amount features were scaled using StandardScaler to bring them to a similar scale as the PCA-transformed features.
Class Imbalance Handling: SMOTE was applied to the training data (X_train, y_train) to oversample the minority class (fraudulent transactions). This helps the models learn better patterns from the underrepresented class.
Model Training
Two classification models were trained and evaluated:

1. Logistic Regression
A basic linear model used as a baseline.
Trained on the SMOTE-resampled training data.
2. XGBoost Classifier
A powerful gradient boosting algorithm known for its performance.
Trained on the SMOTE-resampled training data.
Hyperparameters:
n_estimators: 300
max_depth: 6
learning_rate: 0.05
subsample: 0.8
colsample_bytree: 0.8
scale_pos_weight: 1 (since SMOTE already balanced the classes)
eval_metric: 'logloss'
random_state: 42
Model Evaluation
Models were evaluated on the original, un-resampled test set (X_test, y_test) to reflect real-world performance.

Key Metrics Used:
Accuracy: Overall correctness of the model.
F1-Score: Harmonic mean of precision and recall, particularly useful for imbalanced datasets.
ROC-AUC (Receiver Operating Characteristic - Area Under the Curve): Measures the model's ability to distinguish between classes.
Confusion Matrix: Provides a detailed breakdown of true positives, true negatives, false positives, and false negatives.
Classification Report: Shows precision, recall, f1-score, and support for each class.
Results Comparison:
Metric	Logistic Regression	XGBoost Classifier
Accuracy	0.974	0.998
F1 Score	0.109	0.651
ROC-AUC	0.969	0.980
XGBoost significantly outperforms Logistic Regression, especially in F1-Score, indicating better handling of the minority class.

Real-time Fraud Detection Function
A function real_time_fraud_detection is provided to classify a single transaction and return a prediction along with a risk score (probability).

def real_time_fraud_detection(transaction):
    transaction = np.array(transaction).reshape(1, -1)
    prediction = xgb_model.predict(transaction)[0]
    probability = xgb_model.predict_proba(transaction)[0][1]

    if prediction == 1:
        return f" Fraud Detected (Risk Score: {probability:.2f})"
    else:
        return f" Legitimate Transaction (Risk Score: {probability:.2f})"
