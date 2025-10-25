# train_model.py
"""
Train Phishing Email Detection model using Kaggle dataset:
https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset
"""

import os
import joblib
import pandas as pd
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# ---------------------------------------------------------------------
# 1. Download dataset from Kaggle via kagglehub
# ---------------------------------------------------------------------
print("📦 Downloading dataset from Kaggle...")
dataset_path = kagglehub.dataset_download("naserabdullahalam/phishing-email-dataset")
print("✅ Dataset downloaded to:", dataset_path)

# ---------------------------------------------------------------------
# 2. Load the dataset (adjust CSV filename if needed)
# ---------------------------------------------------------------------
csv_files = [f for f in os.listdir(dataset_path) if f.endswith(".csv")]
if not csv_files:
    raise FileNotFoundError("No CSV file found in downloaded dataset folder.")
csv_path = os.path.join(dataset_path, csv_files[0])

print(f"📄 Loading dataset: {csv_path}")
df = pd.read_csv(csv_path)
print("Dataset columns:", df.columns.tolist())
print("Number of records:", len(df))

# ---------------------------------------------------------------------
# 3. Combine useful columns into one text field (safe string casting)
# ---------------------------------------------------------------------
# Convert all text fields to string before concatenation
for col in ["subject", "body", "urls"]:
    df[col] = df[col].astype(str).fillna("")

df["text"] = df["subject"] + " " + df["body"] + " " + df["urls"]

# Use 'label' as the target column (0 = ham, 1 = phishing)
df["label"] = df["label"].astype(int)

# Drop rows with empty text
df = df[df["text"].str.strip() != ""]
print("✅ Combined text column created. Shape:", df.shape)

# ---------------------------------------------------------------------
# 4. Split into train/test
# ---------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

# ---------------------------------------------------------------------
# 5. Vectorize text (TF-IDF)
# ---------------------------------------------------------------------
vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    stop_words="english",
    ngram_range=(1, 2),
    max_features=10000
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ---------------------------------------------------------------------
# 6. Train Logistic Regression classifier
# ---------------------------------------------------------------------
model = LogisticRegression(max_iter=300, solver="liblinear")
model.fit(X_train_vec, y_train)

# ---------------------------------------------------------------------
# 7. Evaluate model performance
# ---------------------------------------------------------------------
y_pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)
print("\n✅ Model training complete")
print(f"Accuracy: {acc*100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ---------------------------------------------------------------------
# 8. Save model and vectorizer
# ---------------------------------------------------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.joblib")
joblib.dump(vectorizer, "models/vectorizer.joblib")
print("💾 Model and vectorizer saved in /models directory")








# # train_model.py
# """
# Train Phishing Email Detection model using Kaggle dataset:
# https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset
# """

# import os
# import joblib
# import pandas as pd
# import kagglehub
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report, accuracy_score

# # ---------------------------------------------------------------------
# # 1. Download dataset from Kaggle via kagglehub
# # ---------------------------------------------------------------------
# print("📦 Downloading dataset from Kaggle...")
# dataset_path = kagglehub.dataset_download("naserabdullahalam/phishing-email-dataset")
# print("✅ Dataset downloaded to:", dataset_path)

# # ---------------------------------------------------------------------
# # 2. Load the dataset (adjust CSV filename if needed)
# # ---------------------------------------------------------------------
# # The dataset folder may contain "Phishing_Email.csv" or similar
# csv_files = [f for f in os.listdir(dataset_path) if f.endswith(".csv")]
# if not csv_files:
#     raise FileNotFoundError("No CSV file found in downloaded dataset folder.")
# csv_path = os.path.join(dataset_path, csv_files[0])

# print(f"📄 Loading dataset: {csv_path}")
# df = pd.read_csv(csv_path)

# print("Dataset columns:", df.columns.tolist())
# print("Number of records:", len(df))

# # ---------------------------------------------------------------------
# # 3. Clean and prepare text + label
# # ---------------------------------------------------------------------
# # The dataset usually has columns like: 'Email Text', 'Label' or similar
# # Adjust column names if necessary:
# possible_text_cols = ["Email Text", "text", "Email", "email_text", "Body"]
# possible_label_cols = ["Label", "label", "Target", "Phishing"]

# text_col = next((c for c in possible_text_cols if c in df.columns), None)
# label_col = next((c for c in possible_label_cols if c in df.columns), None)

# if not text_col or not label_col:
#     raise ValueError(f"Could not find text/label columns in {df.columns.tolist()}")

# df = df[[text_col, label_col]].dropna()
# df.rename(columns={text_col: "text", label_col: "label"}, inplace=True)

# # Normalize label (0 = legit, 1 = phishing)
# df["label"] = df["label"].astype(str).str.lower().replace({
#     "phishing": 1, "spam": 1, "1": 1,
#     "legit": 0, "ham": 0, "0": 0
# }).astype(int)

# print(df["label"].value_counts())

# # ---------------------------------------------------------------------
# # 4. Split into train/test
# # ---------------------------------------------------------------------
# X_train, X_test, y_train, y_test = train_test_split(
#     df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
# )

# # ---------------------------------------------------------------------
# # 5. Vectorize text (TF-IDF)
# # ---------------------------------------------------------------------
# vectorizer = TfidfVectorizer(
#     sublinear_tf=True,
#     stop_words="english",
#     ngram_range=(1, 2),
#     max_features=10000
# )
# X_train_vec = vectorizer.fit_transform(X_train)
# X_test_vec = vectorizer.transform(X_test)

# # ---------------------------------------------------------------------
# # 6. Train Logistic Regression classifier
# # ---------------------------------------------------------------------
# model = LogisticRegression(max_iter=300, solver="liblinear")
# model.fit(X_train_vec, y_train)

# # ---------------------------------------------------------------------
# # 7. Evaluate model performance
# # ---------------------------------------------------------------------
# y_pred = model.predict(X_test_vec)
# acc = accuracy_score(y_test, y_pred)
# print("\n✅ Model training complete")
# print(f"Accuracy: {acc*100:.2f}%")
# print("\nClassification Report:\n", classification_report(y_test, y_pred))

# # ---------------------------------------------------------------------
# # 8. Save model and vectorizer
# # ---------------------------------------------------------------------
# os.makedirs("models", exist_ok=True)
# joblib.dump(model, "models/model.joblib")
# joblib.dump(vectorizer, "models/vectorizer.joblib")
# print("💾 Model and vectorizer saved in /models directory")

