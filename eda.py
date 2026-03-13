import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re


def main():
    def sanitize_filename(name):
        return re.sub(r'[<>:"/\\|?*]', '_', name)

    INPUT_CSV = "Data_working_final/merged_all_students.csv"
    EDA_OUTPUT_DIR = "Data_working_final/EDA"
    CATEGORICAL_COLS = ['Weather', 'Time of Day', 'Season', 'Activity', 'Mood/Emotion','Country']

    os.makedirs(EDA_OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)

    missing_counts = df[CATEGORICAL_COLS].isnull().sum()
    print("Missing values per column:")
    print(missing_counts)
    # missing_counts.to_csv(os.path.join(EDA_OUTPUT_DIR, "missing_values.csv"))

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            plt.figure(figsize=(10,6))
            counts = df[col].value_counts().head(40)
            percentages = counts / counts.sum() * 100

            ax = sns.barplot(
            x=counts.values,
            y=counts.index,
            hue=counts.index,
            palette="viridis",
            dodge=False,
            legend=False
        )
            plt.xlabel("Count")
            plt.ylabel(col)
            plt.title(f"Class Distribution (Raw) - {col}")

            for i, (count, perc) in enumerate(zip(counts.values, percentages.values)):
                ax.text(count + max(counts.values)*0.01, i, f"{perc:.1f}%", va='center')

            plt.tight_layout()
            plt.savefig(os.path.join(EDA_OUTPUT_DIR, f"{sanitize_filename(col)}_distribution.png"))
            plt.close()
            # counts.to_csv(os.path.join(EDA_OUTPUT_DIR, f"{sanitize_filename(col)}_counts.csv"))

    CATEGORICAL = [col for col in CATEGORICAL_COLS if col != 'Country']
    for i in range(len(CATEGORICAL)):
        for j in range(i+1, len(CATEGORICAL)):
            col1 = CATEGORICAL[i]
            col2 = CATEGORICAL[j]
            if col1 in df.columns and col2 in df.columns:
                ct = pd.crosstab(df[col1], df[col2])
                # ct.to_csv(os.path.join(EDA_OUTPUT_DIR, f"{sanitize_filename(col1)}_{sanitize_filename(col2)}_crosstab.csv"))

                plt.figure(figsize=(10,8))
                sns.heatmap(ct, annot=True, fmt="d", cmap="YlGnBu")
                plt.title(f"{col1} vs {col2} (Raw)")
                plt.tight_layout()
                plt.savefig(os.path.join(EDA_OUTPUT_DIR, f"{sanitize_filename(col1)}_{sanitize_filename(col2)}_heatmap.png"))
                plt.close()

    if 'Description' in df.columns:
        df['desc_length'] = df['Description'].astype(str).str.len()
        plt.figure(figsize=(10,5))
        sns.histplot(df['desc_length'], bins=50, kde=True, color="skyblue")
        plt.xlabel("Description Length")
        plt.ylabel("Count")
        plt.title("Description Length Distribution (Raw)")
        plt.tight_layout()
        plt.savefig(os.path.join(EDA_OUTPUT_DIR, "description_length.png"))
        plt.close()

        desc_stats = df['desc_length'].describe()
        print("\nDescription Length Statistics:")
        print(desc_stats)
        # desc_stats.to_csv(os.path.join(EDA_OUTPUT_DIR, "description_length_stats.csv"))


if __name__ == "__main__":
    main()