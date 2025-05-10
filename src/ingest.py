import pandas as pd
import numpy as np

def initial_ingest(DATA_PATH):
    '''
    Ingest the data from the CSV file and clean it up a bit.
    '''
    df = pd.read_csv(DATA_PATH, engine="python", encoding="utf-8")

    # Sort issues with separators in the company description
    df["company_description"] = df["company_description"].apply(clean_separators)

    # Drop the unnamed column
    df.drop(columns=["Unnamed: 0"], inplace=True)
    
    # Fix the issue with the broken description for company id 756111
    df = fix_756111_issue(df)

    # Force correct types
    df["company_description"] = df["company_description"].astype(str)
    df["source"] = df["source"].astype(str)
    df["is_edited"] = df["is_edited"].astype(int)
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df["id"] = df["id"].astype(int)

    return df

def read_preprocessed_data(PREPROCESSED_PATH):
    '''
    Read the preprocessed data, if it exists
    '''
    try:
        df = pd.read_csv(PREPROCESSED_PATH, engine="python", encoding="utf-8")
        return df
    except FileNotFoundError:
        return None

def read_processed_data(PROCESSED_PATH):
    '''
    Read the processed data, if it exists
    '''
    try:
        df = pd.read_csv(PROCESSED_PATH, engine="python", encoding="utf-8")
        return df
    except FileNotFoundError:
        return None

def read_embeddings(EMBEDDINGS_PATH):
    '''
    Read the embeddings, if they exist
    '''
    try:
        embeddings = np.load(EMBEDDINGS_PATH)
        return embeddings
    except FileNotFoundError:
        return None

def clean_separators(text):
    '''
    Clean the text by removing unwanted separators and replacing them with spaces.
    '''
    return str(text).replace('\u2028', ' ').replace('\u2029', ' ').replace('\u0085', ' ')

def fix_756111_issue(df):
    '''
    Fix the issue with the broken description for company id 756111.
    The description is split across multiple rows, and we need to combine them into one.
    The rows with the broken description are from 367 to 381 (inclusive)
    '''
    broken_rows = df.loc[367:381].copy()
    correct_id = broken_rows.iloc[0]["id"]

    combined_desc_parts = broken_rows["company_description"].dropna().astype(str).tolist()
    combined_desc_parts += broken_rows["id"][1:].dropna().astype(str).tolist()  # skip first 'id' (which is real ID)
    combined_description = " ".join(combined_desc_parts)

    corrected_row = {
        "id": correct_id,
        "company_description": combined_description,
        "source": df["source"].mode()[0], # since no other null values, assume it did have a source value
        "is_edited": 0, # since no other null values, assume is_edited = 0
        "created_at": broken_rows.iloc[-1]["source"], # use the last source value that was in the wrong place
    }

    df_cleaned = df.drop(index=range(367, 382))
    df_cleaned = pd.concat([df_cleaned, pd.DataFrame([corrected_row])], ignore_index=True)

    return df_cleaned