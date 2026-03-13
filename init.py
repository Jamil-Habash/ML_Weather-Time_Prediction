import os
import pandas as pd
from collections import defaultdict
import re
import json
import unicodedata


DATA_DIR = "Data"
OUT_DIR = "Data_working_final"
os.makedirs(OUT_DIR, exist_ok=True)

features = ["Image URL", "Description", "Country", "Weather",
            "Time of Day", "Season", "Activity", "Mood/Emotion"]

possible_times = ["Morning", "Afternoon", "Evening", "Night"]
possible_weather = ["Sunny", "Cloudy", "Rainy", "Snowy",
                    "Windy", "Clear", "Not Clear", "Partly Cloudy"]


def find_folders_with_multiple_files(data_dir):
    result = []
    for folder_name in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder_name)
        if os.path.isdir(folder_path):
            files = [f for f in os.listdir(folder_path)
                     if os.path.isfile(os.path.join(folder_path, f))]
            if len(files) > 1:
                result.append((folder_name, files))

    print("Folders with multiple files:")
    for folder, files in result:
        print(folder, "->", files)


def find_unreadable_files(data_dir):
    for folder_name in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        for f in os.listdir(folder_path):
            if f.lower().endswith(".csv"):
                try:
                    pd.read_csv(os.path.join(folder_path, f))
                except Exception as e:
                    print("Unreadable:", f, e)


def compile_mapping(raw_map):
    compiled = []
    for key, val in raw_map.items():
        tokens = set(normalize_text(key).split())
        if tokens:
            compiled.append((tokens, val))

    compiled.sort(key=lambda x: len(x[0]), reverse=True)
    return compiled


def apply_mapping(x, compiled_map, default=None, preserve_original=False):
    raw = x
    x = normalize_text(x)
    if not x:
        return default if not preserve_original else None

    tokens = set(x.split())
    joined = " ".join(tokens)

    for key_tokens, canonical in compiled_map:
        if key_tokens.issubset(tokens):
            return canonical

        phrase = " ".join(key_tokens)
        if phrase in joined:
            return canonical

    if preserve_original:
        return x.upper()

    return default


def build_columns_by_index_map(data_dir):
    index_map = defaultdict(set)

    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        for f in os.listdir(folder_path):
            if not f.lower().endswith(".csv"):
                continue

            df = pd.read_csv(os.path.join(folder_path, f))

            for idx, col in enumerate(df.columns):
                index_map[idx].add(str(col).strip())

    for idx, names in index_map.items():
        print(f"Index {idx}: {sorted(names)}")


def remove_Extra_rows(df):
    df = df.replace("", pd.NA)
    df = df.dropna(how="all")
    df = df.dropna(axis=1, how="all")
    return df


def normalize_text(x):
    if pd.isna(x):
        return ""

    x = str(x)

    x = unicodedata.normalize("NFKD", x)
    x = x.encode("ascii", "ignore").decode("ascii")
    x = x.lower()
    x = re.sub(r"\(.*?\)", " ", x)
    x = re.sub(r"[^a-z]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()

    return x


def standardize_classes(df):
    with open("mappings.json", "r", encoding="utf-8") as f:
        mappings = json.load(f)

    compiled_maps = {col: compile_mapping(mappings[col]) for col in mappings}

    for col, cmap in compiled_maps.items():
        if col not in df.columns:
            continue

        if col == "Country":
            df[col] = df[col].apply(lambda x: apply_mapping(x, cmap, preserve_original=True))
        else:
            df[col] = df[col].apply(lambda x: apply_mapping(x, cmap, default="NOT CLEAR"))

    return df


def is_header_row(row, features):
    row_values = [re.sub(r"[^a-z0-9]", "", str(x).strip().lower()) for x in row]

    feature_values = [f.lower() for f in features]
    return any(rv in feature_values for rv in row_values)


def build_merged_dataset(data_dir, out_dir=OUT_DIR, features=features):
    all_dfs = []

    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        for f in os.listdir(folder_path):
            if f.lower().endswith(".csv"):

                df = pd.read_csv(os.path.join(folder_path, f), header=None)

                df = remove_Extra_rows(df)
                print(f"Loading: {os.path.join(folder_path, f)} with shape {df.shape}")
                if len(df) > 1 and is_header_row(df.iloc[0], features):
                    df = df.iloc[1:].reset_index(drop=True)
                all_dfs.append(df)

    merged = pd.concat(all_dfs, ignore_index=True)

    rename_map = {i: features[i] for i in range(min(len(features), merged.shape[1]))}
    merged = merged.rename(columns=rename_map)

    merged = merged.dropna(how="all")
    merged = merged.dropna(axis=1, how="all")

    merged = standardize_classes(merged)

    if {"Weather", "Time of Day"}.issubset(merged.columns):
        wt_df = merged[["Image URL", "Weather", "Time of Day"]].copy()
        wt_path = os.path.join(out_dir, "merged_weather_time.csv")
        wt_df.to_csv(wt_path, index=False)

    full_path = os.path.join(out_dir, "merged_all_students.csv")
    merged.to_csv(full_path, index=False)

    print("Saved:", full_path)


def main():
    find_folders_with_multiple_files(DATA_DIR)
    find_unreadable_files(DATA_DIR)
    build_columns_by_index_map(DATA_DIR)
    print("\nBuilding merged dataset (NO original files modified)...")
    build_merged_dataset(DATA_DIR)


if __name__ == "__main__":
    main()