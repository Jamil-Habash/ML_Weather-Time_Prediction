import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score


def prepare_data(
        metadata_csv,
        image_dir,
        target_col,
        image_size=(64, 64),
        test_size=0.2,
        random_state=42
):
    df = pd.read_csv(metadata_csv)

    X = []
    y = []

    for _, row in df.iterrows():
        img_path = os.path.join(image_dir, row["image_filename"])

        if not os.path.exists(img_path):
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize(image_size)
        except Exception:
            print(f"Error loading image: {img_path}")
            continue

        img_array = np.array(img).flatten()
        X.append(img_array)
        y.append(row[target_col])

    if len(X) == 0:
        raise RuntimeError("No images were loaded. Check image paths and filenames.")

    X = np.array(X)
    y = np.array(y)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test, label_encoder


def train_and_evaluate_knn(
        X_train,
        y_train,
        X_test,
        y_test,
        distance="euclidean"
):
    for k in [1, 3]:
        knn = KNeighborsClassifier(
            n_neighbors=k,
            metric=distance
        )

        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")

        print(f"KNN (k={k}, distance={distance}) Accuracy: {acc:.4f}")
        print(f"KNN (k={k}) F1-score: {f1:.4f}")


def main():
    METADATA_CSV = "Data_working_final/metadata.csv"
    IMAGE_DIR = "Data_working_final/images"
    for TARGET_LABEL in ["Weather", "Time of Day"]:
        print(f"\nEvaluating for {TARGET_LABEL}\n")

        X_train, X_test, y_train, y_test, _ = prepare_data(
            metadata_csv=METADATA_CSV,
            image_dir=IMAGE_DIR,
            target_col=TARGET_LABEL
        )

        train_and_evaluate_knn(
            X_train,
            y_train,
            X_test,
            y_test,
            distance="euclidean"
        )


if __name__ == "__main__":
    main()
