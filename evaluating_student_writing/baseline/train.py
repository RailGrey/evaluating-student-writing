from pathlib import Path

import joblib
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

from evaluating_student_writing.baseline.data import (
    build_sentence_dataset,
    build_tfidf,
    split_dataset,
)
from evaluating_student_writing.baseline.utils import CLASSES

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models" / "baseline"


def train() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = PROJECT_ROOT / "data" / "train.csv"
    essays_dir = PROJECT_ROOT / "data" / "train"

    print("Building sentence dataset...")
    df = build_sentence_dataset(csv_path, essays_dir)
    print(f"Total sentences: {len(df)}")
    print(f"Label distribution:\n{df['label'].value_counts()}")

    train_df, val_df = split_dataset(df)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")

    print("Building TF-IDF features...")
    vectorizer, X_train, X_val = build_tfidf(
        train_df["sentence_text"], val_df["sentence_text"]
    )

    label_to_idx = {label: idx for idx, label in enumerate(CLASSES)}
    y_train = train_df["label"].map(label_to_idx)
    y_val = val_df["label"].map(label_to_idx)

    print("Training XGBoost...")
    model = XGBClassifier(
        n_estimators=10,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        num_class=len(CLASSES),
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=20,
    )

    y_pred = model.predict(X_val)
    print("\nValidation Classification Report:")
    print(classification_report(y_val, y_pred, target_names=CLASSES))

    joblib.dump(model, MODEL_DIR / "xgb_model.joblib")
    joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer.joblib")
    joblib.dump(label_to_idx, MODEL_DIR / "label_to_idx.joblib")
    print(f"\nModel saved to {MODEL_DIR}")


if __name__ == "__main__":
    train()
