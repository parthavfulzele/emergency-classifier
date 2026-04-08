"""
clean_data.py — Data Cleaning Pipeline for Emergency Text Classification Dataset
Author: Parthav Fulzele
Repository: https://github.com/parthavfulzele/emergency-classifier

This script performs all cleaning steps on the raw dataset (dataset.csv)
and outputs a cleaned version (dataset_cleaned.csv) along with a summary
of every transformation applied.
"""

import re
import pandas as pd
import numpy as np


def load_raw(path: str) -> pd.DataFrame:
    """Load the raw CSV and print initial shape."""
    df = pd.read_csv(path)
    print(f"[RAW] Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"       Columns: {list(df.columns)}")
    print(f"       Dtypes : text={df['text'].dtype}, label={df['label'].dtype}")
    return df


def check_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Identify and drop rows with missing values."""
    missing = df.isnull().sum()
    print(f"\n[MISSING VALUES]")
    print(f"  text : {missing['text']}")
    print(f"  label: {missing['label']}")
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    print(f"  Dropped {before - len(df)} rows with missing values")
    return df


def check_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Detect and remove exact-duplicate rows."""
    exact_dupes = df.duplicated().sum()
    text_dupes = df['text'].duplicated().sum()
    print(f"\n[DUPLICATES]")
    print(f"  Exact duplicate rows : {exact_dupes}")
    print(f"  Duplicate texts only : {text_dupes}")
    before = len(df)
    df = df.drop_duplicates(subset='text', keep='first').reset_index(drop=True)
    print(f"  Removed {before - len(df)} duplicate-text rows")
    return df


def fix_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure correct dtypes: text as string, label as int (0 or 1)."""
    print(f"\n[DATA TYPES]")
    df['text'] = df['text'].astype(str)
    df['label'] = df['label'].astype(int)
    invalid_labels = df[~df['label'].isin([0, 1])]
    print(f"  Invalid labels (not 0/1): {len(invalid_labels)}")
    if len(invalid_labels) > 0:
        df = df[df['label'].isin([0, 1])].reset_index(drop=True)
        print(f"  Removed {len(invalid_labels)} rows with invalid labels")
    else:
        print("  All labels valid (0 or 1)")
    return df


def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize whitespace, strip leading/trailing spaces,
       and lowercase all text for consistency."""
    print(f"\n[TEXT NORMALIZATION]")

    # Leading/trailing whitespace
    ws_issues = (df['text'] != df['text'].str.strip()).sum()
    print(f"  Rows with leading/trailing whitespace: {ws_issues}")
    df['text'] = df['text'].str.strip()

    # Double spaces
    dbl = df['text'].str.contains(r'  ').sum()
    print(f"  Rows with double spaces: {dbl}")
    df['text'] = df['text'].str.replace(r'\s+', ' ', regex=True)

    # Casing — lowercase everything for model consistency
    upper_start = df['text'].str[0].str.isupper().sum()
    print(f"  Rows starting uppercase (before lowering): {upper_start}/{len(df)}")
    df['text'] = df['text'].str.lower()
    print("  All text lowercased")

    return df


def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Identify length-based outliers (very short or very long texts)."""
    print(f"\n[OUTLIER ANALYSIS]")
    df['_word_count'] = df['text'].str.split().str.len()
    df['_char_count'] = df['text'].str.len()

    q1 = df['_word_count'].quantile(0.25)
    q3 = df['_word_count'].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = df[(df['_word_count'] < lower) | (df['_word_count'] > upper)]
    print(f"  Word count IQR: Q1={q1}, Q3={q3}, IQR={iqr}")
    print(f"  Outlier bounds: <{lower:.1f} or >{upper:.1f} words")
    print(f"  Outlier rows: {len(outliers)}")
    if len(outliers) > 0:
        for _, row in outliers.iterrows():
            print(f"    [{row['label']}] ({row['_word_count']} words) \"{row['text'][:60]}...\"")

    # We keep outliers but flag them — single-word entries are realistic
    # voice inputs (e.g., someone yelling "fire!" or "help")
    print("  Decision: KEEP outliers — short texts are realistic for voice reports")

    df = df.drop(columns=['_word_count', '_char_count'])
    return df


def summary(raw_df: pd.DataFrame, clean_df: pd.DataFrame):
    """Print before/after comparison."""
    print(f"\n{'='*55}")
    print(f"  BEFORE/AFTER SUMMARY")
    print(f"{'='*55}")
    print(f"  {'Metric':<30} {'Before':>10} {'After':>10}")
    print(f"  {'-'*50}")
    print(f"  {'Rows':<30} {len(raw_df):>10} {len(clean_df):>10}")
    print(f"  {'Columns':<30} {raw_df.shape[1]:>10} {clean_df.shape[1]:>10}")
    print(f"  {'Missing (text)':<30} {raw_df['text'].isnull().sum():>10} {clean_df['text'].isnull().sum():>10}")
    print(f"  {'Missing (label)':<30} {raw_df['label'].isnull().sum():>10} {clean_df['label'].isnull().sum():>10}")
    print(f"  {'Duplicates':<30} {raw_df.duplicated().sum():>10} {clean_df.duplicated().sum():>10}")
    print(f"  {'Emergency (label=1)':<30} {(raw_df['label']==1).sum():>10} {(clean_df['label']==1).sum():>10}")
    print(f"  {'Non-Emergency (label=0)':<30} {(raw_df['label']==0).sum():>10} {(clean_df['label']==0).sum():>10}")
    print(f"{'='*55}")


def main():
    RAW_PATH = "dataset.csv"
    CLEAN_PATH = "dataset_cleaned.csv"

    raw_df = load_raw(RAW_PATH)
    df = raw_df.copy()

    df = check_missing(df)
    df = check_duplicates(df)
    df = fix_data_types(df)
    df = normalize_text(df)
    df = handle_outliers(df)

    summary(raw_df, df)

    df.to_csv(CLEAN_PATH, index=False)
    print(f"\nCleaned dataset saved to {CLEAN_PATH}")


if __name__ == "__main__":
    main()
