import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

print("Loading dataset...")
df = pd.read_csv('dementia_patients_health_data.csv')

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

target_column = 'Dementia'

if target_column not in df.columns:
    print(f"\n❌ Target column '{target_column}' not found!")
    print(f"Available columns: {df.columns.tolist()}")
    exit()

X = df.drop(target_column, axis=1)
y = df[target_column]

print(f"\nTarget distribution:\n{y.value_counts()}")

print(f"\nMissing values per column:")
missing = X.isnull().sum()
if missing.sum() > 0:
    print(missing[missing > 0])
    print(f"\nTotal missing values: {missing.sum()}")
else:
    print("No missing values found")

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"\nCategorical columns ({len(categorical_cols)}): {categorical_cols}")
print(f"Numerical columns ({len(numerical_cols)}): {numerical_cols}")

for col in numerical_cols:
    if X[col].isnull().sum() > 0:
        median_val = X[col].median()
        X[col].fillna(median_val, inplace=True)
        print(f"✓ Filled {X[col].isnull().sum()} missing values in {col} with median: {median_val}")

label_encoders = {}
for col in categorical_cols:
    if X[col].isnull().sum() > 0:
        mode_val = X[col].mode()[0]
        X[col].fillna(mode_val, inplace=True)
        print(f"✓ Filled missing values in {col} with mode: {mode_val}")
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le
    print(f"✓ Encoded {col}: {len(le.classes_)} classes")

print(f"\nMissing values after cleaning: {X.isnull().sum().sum()}")

feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

print("\nTraining Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n{'='*60}")
print(f"Model Accuracy: {accuracy:.4f}")
print(f"{'='*60}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nSaving model files...")

with open('dementia_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✓ Saved: dementia_model.pkl")

with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
print("✓ Saved: label_encoders.pkl")

with open('feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)
print("✓ Saved: feature_names.pkl")

print("\n" + "="*60)
print("✅ MODEL TRAINING COMPLETE!")
print("="*60)
print("\nGenerated files:")
print("  1. dementia_model.pkl")
print("  2. label_encoders.pkl")
print("  3. feature_names.pkl")
print("\nYou can now ush these files to GitHub and deploy to Render.")
print("="*60)