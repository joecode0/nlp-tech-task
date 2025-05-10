
import os
import argparse
import time
from src import ingest, preprocess, feature_engineering, structure, visualise

OUTPUT_DIR = "output"
DATA_PATH = "data/2025_data_to_explore.csv"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
PREPROCESSED_PATH = os.path.join(OUTPUT_DIR, "preprocessed_data.csv")
PROCESSED_PATH = os.path.join(OUTPUT_DIR, "processed_data.csv")
TFIDF_MATRIX_PATH = os.path.join(OUTPUT_DIR, "tfidf_matrix.npz")
TFIDF_TERMS_PATH = os.path.join(OUTPUT_DIR, "tfidf_terms.csv")
EMBEDDINGS_PATH = os.path.join(OUTPUT_DIR, "sentence_embeddings.npy")

def run_pipeline(run_stage="all"):
    df = None
    embeddings = None
    tfidf_matrix = None
    tfidf_terms = None

    if run_stage in ["all", "ingest"]:
        print("Step 1: Ingesting raw data...")
        df = ingest.initial_ingest(DATA_PATH)

    if run_stage in ["all", "preprocess"]:
        print("Step 2: Preprocessing text...")
        if df is None:
            df = ingest.initial_ingest(DATA_PATH)
            print("Ingested data from CSV file as the DataFrame was not found.")
        t1 = time.time()
        df = preprocess.preprocess_data(df)
        t2 = time.time()
        print(f"Preprocessing took {t2 - t1:.2f} seconds.")
        # Save the preprocessed data to a CSV file
        df.to_csv(PREPROCESSED_PATH, index=False)
        print(f"Preprocessed data saved to {PREPROCESSED_PATH}.")

    if run_stage in ["all", "features", "features_structure"]:
        print("Step 3: Feature engineering...")
        # Check if preprocessed data exists, if not, get or generate it
        if df is None:
            df = ingest.read_preprocessed_data(PREPROCESSED_PATH)
            if df is None:
                print("Step 1: Ingesting raw data...")
                df = ingest.initial_ingest(DATA_PATH)
                print("Step 2: Preprocessing text...")
                df = preprocess.preprocess_data(df)

        t1 = time.time()
        features = feature_engineering.feature_engineering(df)
        t2 = time.time()
        print(f"Feature Engineering took {t2 - t1:.2f} seconds.")

    if run_stage in ["all", "features_structure"]:
        if features is None:
            print("Pipeline aborted: No features to structure.")
            return
        print("Step 4: Structuring & saving final datasets...")
        structure.save_structured_data(
                    features=features,
                    processed_path=PROCESSED_PATH,
                    tfidf_matrix_path=TFIDF_MATRIX_PATH,
                    tfidf_terms_path=TFIDF_TERMS_PATH,
                    embeddings_path=EMBEDDINGS_PATH
        )

    if run_stage in ["all", "visualise"]:
        print("Step 5: Generating visualisations...")
        features = {}
        features["df"] = ingest.read_processed_data(PROCESSED_PATH)
        features["embeddings"] = ingest.read_embeddings(EMBEDDINGS_PATH)
        t1 = time.time()
        visualise.generate_visualisations(
            df=features["df"],
            embeddings=features["embeddings"],
            PLOTS_DIR=PLOTS_DIR
        )
        t2 = time.time()
        print(f"Visualisation generation and saving took {t2 - t1:.2f} seconds.")

    print("Pipeline completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pipeline stages for the NLP technical assessment.")
    parser.add_argument("--run", type=str, choices=["all", "ingest", "preprocess", "features", "features_structure", "visualise"],
                        default="all", help="Pipeline stage to run")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Step 0: Setting up...")
    run_pipeline(args.run)
