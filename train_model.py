import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# Veri setini oku
# -----------------------------

df = pd.read_csv("Food_Delivery_Times.csv")

# -----------------------------
# Eksik veriler
# -----------------------------

df["Weather"] = df["Weather"].fillna(df["Weather"].mode()[0])
df["Traffic_Level"] = df["Traffic_Level"].fillna(df["Traffic_Level"].mode()[0])
df["Time_of_Day"] = df["Time_of_Day"].fillna(df["Time_of_Day"].mode()[0])

df["Courier_Experience_yrs"] = (
    df["Courier_Experience_yrs"]
    .fillna(df["Courier_Experience_yrs"].median())
)

# -----------------------------
# Hedef değişken
# -----------------------------

df["delivery_risk"] = (
    df["Delivery_Time_min"] > 60
).astype(int)

# -----------------------------
# X ve y
# -----------------------------

X = df.drop(
    columns=[
        "Order_ID",
        "Delivery_Time_min",
        "delivery_risk"
    ]
)

y = df["delivery_risk"]

# -----------------------------
# Kategorik kolonlar
# -----------------------------

categorical_cols = [
    "Weather",
    "Traffic_Level",
    "Time_of_Day",
    "Vehicle_Type"
]

# -----------------------------
# One Hot Encoding
# -----------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_cols
        )
    ],
    remainder="passthrough"
)

# -----------------------------
# Train Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Logistic Regression
# -----------------------------

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)

preds = model.predict(X_test)

print("\nAccuracy:")
print(accuracy_score(y_test, preds))

print("\nClassification Report:")
print(classification_report(y_test, preds))

print("\n")
print("=" * 50)
print("RANDOM FOREST")
print("=" * 50)

rf_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ))
])

rf_model.fit(X_train, y_train)

rf_preds = rf_model.predict(X_test)

print("\nAccuracy:")
print(accuracy_score(y_test, rf_preds))

print("\nClassification Report:")
print(classification_report(y_test, rf_preds))

# -----------------------------
# En iyi modeli kaydet
# -----------------------------

joblib.dump(model, "best_delivery_model.pkl")

print("\nEn iyi model Logistic Regression olarak kaydedildi:")
print("best_delivery_model.pkl")