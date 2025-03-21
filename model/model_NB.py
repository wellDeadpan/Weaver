from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# Features and labels
X = df_final.drop("heart_failure", axis=1)
y = df_final["heart_failure"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Manually set priors — treat HF and non-HF equally
# [P(non-HF), P(HF)]
nb_model = GaussianNB(priors=[0.5, 0.5])
nb_model.fit(X_train, y_train)

# Predict classes
y_pred = nb_model.predict(X_test)

# Predict probabilities (for threshold tuning later)
y_proba = nb_model.predict_proba(X_test)[:, 1]

from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))


# Adjust threshold
threshold = 0.3
y_pred_thresh = (y_proba > threshold).astype(int)

print(f"Classification Report (Threshold={threshold}):")
print(classification_report(y_test, y_pred_thresh))


