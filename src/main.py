import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC  
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


def SVM_LINEAR(X_train, X_test, y_train):
    """Trains and predicts using a Support Vector Machine with a linear kernel."""
    svm_linear = SVC(kernel='linear', C=1.0, random_state=42, 
                     probability=True, cache_size=500)
    svm_linear.fit(X_train, y_train)
    y_pred = svm_linear.predict(X_test)
    y_prob = svm_linear.predict_proba(X_test)[:, 1]
    return y_pred, y_prob

def SVM_KERNEL_POLYNOMIAL(X_train, X_test, y_train):
    """Trains and predicts using a Support Vector Machine with a polynomial kernel."""
    svm_poly = SVC(kernel='poly', degree=3, C=1.0, random_state=42, 
                   probability=True, cache_size=500)
    svm_poly.fit(X_train, y_train)
    y_pred = svm_poly.predict(X_test)
    y_prob = svm_poly.predict_proba(X_test)[:, 1]
    return y_pred, y_prob

def SVM_KERNEL_EXPONENTIELLE(X_train, X_test, y_train):
    """Trains and predicts using a Support Vector Machine with an RBF (Exponential) kernel."""
    svm_rbf = SVC(kernel='rbf', gamma='scale', C=1.0, random_state=42, 
                  probability=True, cache_size=500)
    svm_rbf.fit(X_train, y_train)
    y_pred = svm_rbf.predict(X_test)
    y_prob = svm_rbf.predict_proba(X_test)[:, 1]
    return y_pred, y_prob

def logisticRegression_model(X_train, X_test, y_train):
    """Trains and predicts using a Logistic Regression model."""
    model = LogisticRegression(max_iter=10000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return y_pred, y_prob


results = []

def collect_metrics(model_name, y_pred, y_prob, y_test):
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    results.append({
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "ROC AUC": roc_auc_score(y_test, y_prob),
        "F1-Score": f1_score(y_test, y_pred),
        "Specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
        "Sensitivity": recall_score(y_test, y_pred, zero_division=0),
        "Precision": precision_score(y_test, y_pred, zero_division=0)
    })

# --- Data Preprocessing ---
file_path = 'data/Social_Network_Data.xlsx'
df = pd.read_excel(file_path)

# Map gender categorical data to numerical
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

# Separate features and target
X = df.drop(['UserID', 'Purchased'], axis=1) 
y = df['Purchased']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling for better model performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) 
X_test = scaler.transform(X_test)     


# ========== SVM LINEAR ==========
y_pred_linear , y_prob_linear = SVM_LINEAR(X_train, X_test, y_train)

# ========== SVM KERNEL POLYNOMIAL ==========
y_pred_poly,y_prob_poly = SVM_KERNEL_POLYNOMIAL(X_train, X_test, y_train)

# ========== SVM KERNEL EXPONENTIELLE ==========
y_pred_rbf,y_prob_rbf = SVM_KERNEL_EXPONENTIELLE(X_train, X_test, y_train)

# ========== LOGISTIC REGRESSION ==========
y_pred_regression_L, y_prob_regression_L = logisticRegression_model(X_train, X_test, y_train)


# Collect metrics for all models
collect_metrics("SVM Linear", y_pred_linear, y_prob_linear, y_test)
collect_metrics("SVM Polynomial", y_pred_poly, y_prob_poly, y_test)
collect_metrics("SVM RBF", y_pred_rbf, y_prob_rbf, y_test)
collect_metrics("Logistic Regression", y_pred_regression_L, y_prob_regression_L, y_test)

comparison_df = pd.DataFrame(results)
print("\n========== FINAL COMPARISON ==========\n")
print(comparison_df.sort_values(by="ROC AUC", ascending=False).to_string(index=False))