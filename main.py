import knn as basemodel
import download as dl
import init as initialize
import eda as eda_analysis
import cnn_randomforest as cnn_rf
import analysis as error_analysis


def ask_user(question):
    return input(question + " (y/n): ").strip().lower() in ["y", "yes", "1"]

CATEGS = ['Image URL', 'Description', 'Country', 'Weather', 'Time of Day', 'Season', 'Activity', 'Mood/Emotion']


def main():
    print("\n=== ENCS5341: Assignment 3 ===")

    if ask_user("\nFormat and initialize the dataset"):
        initialize.main()

    if ask_user("\nDownload images and metadata"):
        dl.download_images_with_metadata(
            in_csv="Data_working_final/merged_all_students.csv",
            out_dir="Data_working_final",
            image_url_col="Image URL",
            label_cols=('Description', 'Country', 'Weather', 'Time of Day', 'Season', 'Activity', 'Mood/Emotion')
        )

    if ask_user("\nPerform Exploratory Data Analysis (EDA)"):
        eda_analysis.main()

    if ask_user("\nEvaluate baseline model k-NN"):
        basemodel.main()

    if ask_user("\nRun CNN & Random Forest models"):
        print("\nLearning Rate 0.0005 & RF Trees Count 50:")
        cnn_rf.main(0.0005, 50)
        print("\nLearning Rate 0.001 & RF Trees Count 100:")
        cnn_rf.main(0.001, 100)
        print("\nLearning Rate 0.005 & RF Trees Count 300:")
        cnn_rf.main(0.005, 300)
        print("\nLearning Rate 0.01 & RF Trees Count 500:")
        cnn_rf.main(0.01, 500)

    if ask_user("\nRun Analysis For Best Performing Model"):
        error_analysis.main()


if __name__ == "__main__":
    main()
