
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

def train_model():
    columns = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "species"
    ]

    data = pd.read_csv(
        "iris_data.csv",
        header=None,
        names=columns,
        skip_blank_lines=True
    )

    data = data.dropna(how="all")

    print("Dataset loaded successfully")
    print("Number of records:", len(data))

    X = data[columns[:4]]
    y = data["species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    print("Training model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("Accuracy:", round(accuracy, 4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    joblib.dump(model, "iris_model.pkl")

    metrics = {
        "accuracy": float(accuracy),
        "training_records": len(X_train),
        "testing_records": len(X_test),
        "species_count": int(y.nunique())
    }

    with open("metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    print("Model saved: iris_model.pkl")
    print("Metrics saved: metrics.json")

if __name__ == "__main__":
    train_model()
