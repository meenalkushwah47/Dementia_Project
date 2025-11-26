import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
import pickle

# Load data
data = pd.read_csv("dementia_patients_health_data.csv")

# Columns to impute
cols_to_impute = ["Prescription", "Dosage in mg", "Chronic_Health_Conditions"]

# Create copy for clustering
df_copy = data.copy()

# Encode categorical variables for clustering
encoders = {}
for col in df_copy.columns:
    if df_copy[col].dtype == "object":
        encoders[col] = LabelEncoder()
        df_copy[col] = encoders[col].fit_transform(df_copy[col].astype(str))

# Handle missing values before clustering
df_processed = df_copy.fillna(df_copy.median())

# Perform clustering
kmeans = KMeans(n_clusters=5, random_state=42)
df_copy["cluster"] = kmeans.fit_predict(df_processed)

# Add cluster to original data
data["cluster"] = df_copy["cluster"]

# Impute missing values using cluster-based mode
for col in cols_to_impute:
    for cluster_id in data['cluster'].unique():
        cluster_rows = data[data['cluster'] == cluster_id][col]
        mode_value = cluster_rows.mode()
        if len(mode_value) == 0:
            fill_value = "Unknown"
        else:
            fill_value = mode_value.iloc[0]
        data.loc[(data['cluster'] == cluster_id) & (data[col].isnull()), col] = fill_value

# Drop cluster column
data.drop(columns=["cluster"], inplace=True)

# Encode all categorical variables
data_encoded = data.copy()
label_encoders = {}

for col in data_encoded.select_dtypes(include='object'):
    le = LabelEncoder()
    data_encoded[col] = le.fit_transform(data_encoded[col])
    label_encoders[col] = le

# Prepare features and target
X = data_encoded.drop('Dementia', axis=1)
y = data_encoded['Dementia']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluate model
pred = rf.predict(X_test)
acc = accuracy_score(y_test, pred)
mse = mean_squared_error(y_test, pred)

print(f"Accuracy: {acc}")
print(f"MSE: {mse}")

# Save model and encoders
with open('dementia_model.pkl', 'wb') as f:
    pickle.dump(rf, f)

with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

with open('feature_names.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)

print("Model saved successfully!")