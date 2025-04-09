from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import pickle

from data.DataProcessing import features_df


features_df = features_df.drop(['HFSTART', 'BIRTHDATE'], axis=1)
# Assuming `features_df` is your DataFrame with categorical variables

# List of categorical columns to encode
categorical_columns = ['MARITAL', 'RACE', 'ETHNICITY', 'GENDER']

# Initialize LabelEncoder and OneHotEncoder
label_encoders = {col: LabelEncoder() for col in categorical_columns}
onehot_encoder = OneHotEncoder(sparse_output=False, drop='first')  # drop='first' to avoid multicollinearity

# Apply LabelEncoder to each categorical column
for col in categorical_columns:
    features_df[col] = label_encoders[col].fit_transform(features_df[col])

# Apply OneHotEncoder to the DataFrame
encoded_features = onehot_encoder.fit_transform(features_df[categorical_columns])

# Create a DataFrame with the one-hot encoded columns
encoded_df = pd.DataFrame(encoded_features, columns=onehot_encoder.get_feature_names_out(categorical_columns))

# Drop the original categorical columns and concatenate the one-hot encoded columns
features_df = features_df.drop(categorical_columns, axis=1)
features_df = pd.concat([features_df, encoded_df], axis=1)


# Fill NA with 0 for count columns
features_df.fillna(0, inplace=True)
# Display the updated DataFrame
print(features_df.head())

# Separate features and target
X = features_df.drop(['PATIENT', 'HFYN'], axis=1)
y = features_df['HFYN']



X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Manually set priors — treat HF and non-HF equally
# [P(non-HF), P(HF)]
nb_model = GaussianNB(priors=[0.5, 0.5])
nb_model.fit(X_train, y_train)

# Predict classes
y_pred = nb_model.predict(X_test)

# Predict probabilities (for threshold tuning later)
y_proba = nb_model.predict_proba(X_test)[:, 1]


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




# Save the trained model
with open("model/model_NB.pkl", "wb") as f:
    pickle.dump(nb_model, f)

# Optional: save encoders too if you need them for new data
with open("model/label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

with open("model/onehot_encoder.pkl", "wb") as f:
    pickle.dump(onehot_encoder, f)
