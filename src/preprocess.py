import pandas as pd
import re
from src.preprocess_utils import (
    ascii_ratio, symbol_ratio, digit_ratio, detect_lang, 
    mask_and_extract_all, mask_other, lemmatize_text, 
    advanced_doc_stats
)

def preprocess_data(df):
    """
    Preprocess the input data to prepare for feature engineering
    """
    # Filter the data down to only quality data and add some document stats
    df = initial_quality_filter(df)

    # Clean up the company description strings
    df["cleaned_description"] = df["company_description"].apply(clean_company_description)

    # Add some other document stats
    df = advanced_doc_stats(df)

    # Handle values we want to mask, and add features to track them
    mask_df = df["cleaned_description"].apply(mask_and_extract_all).apply(pd.Series)

    # Put the data back in, replacing the cleaned description with the masked version
    df = pd.concat([df, mask_df], axis=1)

    # Handle other masked values that we don't want to track
    df["masked_description"] = df["masked_description"].apply(mask_other)

    # Lowercase only after masking
    df["masked_description"] = df["masked_description"].str.lower()

    # Lemmatize the cleaned description
    df["lemmatized_description"] = df["masked_description"].apply(lemmatize_text)

    return df

def initial_quality_filter(df):
    '''
    Perform initial cleanup on the DataFrame based on text quality and duplicates/missing values.
    '''
    # Generate additional features for initial filtering
    df["char_count"] = df["company_description"].apply(len).astype(int)
    df["word_count"] = df["company_description"].apply(lambda x: len(x.split())).astype(int)
    df["ascii_ratio"] = df["company_description"].apply(ascii_ratio).astype(float)
    df["symbol_ratio"] = df["company_description"].apply(symbol_ratio).astype(float)
    df["digit_ratio"] = df["company_description"].apply(digit_ratio).astype(float)
    df["language"] = df["company_description"].apply(detect_lang).astype(str)

    # Filter rows based on above thresholds
    df = df[(df["char_count"] >= 40) & (df["word_count"] >= 10) & (df["ascii_ratio"] >= 0.6) & (df["symbol_ratio"] <= 0.15)]
    df = df[df["language"] == "en"]

    # Clean up source column values to website & linkedin
    df["source"] = df["source"].apply(lambda x: "linkedin" if x == "LinkedIn - Reported" else "website")

    # Reset index and clean up
    df.reset_index(drop=True, inplace=True)
    df.drop(columns=["language","ascii_ratio","symbol_ratio","digit_ratio"], inplace=True)

    # Drop any duplicate rows based on the 'id' column
    df.drop_duplicates(subset=["id"], inplace=True)
    
    # Drop any rows with null value in 'company_description'
    df.dropna(subset=["company_description"], inplace=True)

    return df

def clean_company_description(text):
    """
    Clean the company description by removing unwanted characters and formatting.
    """
    # Enforce fixed encoding
    text = str(text).encode("utf-8", "ignore").decode("utf-8", "ignore")

    # Remove any unwanted characters (e.g., newlines, tabs, etc.)
    text = str(text).replace('\n', ' ').replace('\t', ' ').replace('\r', ' ')

    # Remove html
    text = text.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")

    # Remove any non-ASCII characters
    text = ''.join(c for c in text if ord(c) < 128)
    
    # Solve the backslash issue
    text = re.sub(r"\\+", " ", text)

    # Remove any extra spaces
    text = ' '.join(text.strip().split())
    
    return text