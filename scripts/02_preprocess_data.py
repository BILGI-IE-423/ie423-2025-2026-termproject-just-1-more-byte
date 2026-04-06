"""
02_preprocess_data.py
---------------------
IE 423 — Term Project
Script 2: Clean and preprocess the raw MBTI dataset.

This script:
- Loads the raw dataset
- Splits posts into individual entries
- Cleans text (removes URLs, special characters, extra whitespace)
- Creates binary dimension columns (I/E, N/S, T/F, J/P)
- Adds basic linguistic features (word count, post count)
- Saves the cleaned dataset to data/processed/mbti_cleaned.csv
"""

import os
import re
import pandas as pd

# --- File paths ---
RAW_DATA_PATH       = os.path.join("data", "raw", "mbti_1.csv")
PROCESSED_DATA_PATH = os.path.join("data", "processed", "mbti_cleaned.csv")
TABLES_PATH         = os.path.join("outputs", "tables")

os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
os.makedirs(TABLES_PATH, exist_ok=True)

# --- 1. Load raw data ---
print("Loading raw data...")

df = pd.read_csv(RAW_DATA_PATH)
print(f"Loaded {df.shape[0]} rows, columns: {list(df.columns)}")

# --- 2. Standardize column names ---

df.columns = [col.strip().lower() for col in df.columns]
df["type"] = df["type"].str.strip().str.upper()
print(f"\n[OK] Column names standardized: {list(df.columns)}")
print(f"     MBTI type column uppercased. Example: {df['type'].iloc[0]}")

# --- 3. Drop duplicates and missing values ---

initial_rows = len(df)
df.drop_duplicates(inplace=True)
df.dropna(subset=["type", "posts"], inplace=True)
final_rows = len(df)

print(f"\n  Rows before cleaning : {initial_rows}")
print(f"  Rows after cleaning  : {final_rows}")
print(f"  Rows removed         : {initial_rows - final_rows}")

# --- 4. Split posts ---

df["post_list"]  = df["posts"].apply(lambda x: [p.strip() for p in x.split("|||") if p.strip()])
df["post_count"] = df["post_list"].apply(len)

print(f"\n  Mean posts per user : {df['post_count'].mean():.1f}")
print(f"  Min posts per user  : {df['post_count'].min()}")
print(f"  Max posts per user  : {df['post_count'].max()}")

# --- 5. Text cleaning ---
print("\nCleaning text...")

def clean_text(text):
    """Remove URLs, MBTI type mentions, special chars, extra spaces."""
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    # Remove MBTI type mentions (to avoid label leakage)
    mbti_pattern = r"\b(INTJ|INTP|ENTJ|ENTP|INFJ|INFP|ENFJ|ENFP|ISTJ|ISFJ|ESTJ|ESFJ|ISTP|ISFP|ESTP|ESFP)\b"
    text = re.sub(mbti_pattern, "", text, flags=re.IGNORECASE)
    # Remove special characters and digits
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text

df["clean_posts"] = df["posts"].apply(clean_text)
print("\n[OK] Text cleaned: URLs, MBTI mentions, special chars removed.")
print(f"\nSample — Original (first 100 chars) : {df['posts'].iloc[0][:100]}")
print(f"Sample — Cleaned  (first 100 chars) : {df['clean_posts'].iloc[0][:100]}")

# --- 6. Binary dimension columns ---

df["dim_IE"] = df["type"].apply(lambda t: 0 if t[0] == "I" else 1)  # 0=Introvert, 1=Extrovert
df["dim_NS"] = df["type"].apply(lambda t: 0 if t[1] == "N" else 1)  # 0=Intuitive, 1=Sensing
df["dim_TF"] = df["type"].apply(lambda t: 0 if t[2] == "T" else 1)  # 0=Thinking, 1=Feeling
df["dim_JP"] = df["type"].apply(lambda t: 0 if t[3] == "J" else 1)  # 0=Judging, 1=Perceiving

print("\n  Binary columns created:")
print("    dim_IE : 0 = Introvert,  1 = Extrovert")
print("    dim_NS : 0 = Intuitive,  1 = Sensing")
print("    dim_TF : 0 = Thinking,   1 = Feeling")
print("    dim_JP : 0 = Judging,    1 = Perceiving")

for col in ["dim_IE", "dim_NS", "dim_TF", "dim_JP"]:
    counts = df[col].value_counts().sort_index()
    print(f"\n  {col} distribution: {dict(counts)}")

# --- 7. Linguistic features ---

df["word_count"]   = df["clean_posts"].apply(lambda x: len(x.split()))
df["char_count"]   = df["clean_posts"].apply(len)
df["avg_word_len"] = df.apply(
    lambda row: row["char_count"] / row["word_count"] if row["word_count"] > 0 else 0, axis=1
)

print("\n  Linguistic features added:")
print(f"    word_count   — mean: {df['word_count'].mean():.0f}")
print(f"    char_count   — mean: {df['char_count'].mean():.0f}")
print(f"    avg_word_len — mean: {df['avg_word_len'].mean():.2f}")

# --- 8. Save processed dataset ---
print("\nSaving processed dataset...")

save_cols = ["type", "dim_IE", "dim_NS", "dim_TF", "dim_JP",
             "clean_posts", "post_count", "word_count", "char_count", "avg_word_len"]
df[save_cols].to_csv(PROCESSED_DATA_PATH, index=False)
print(f"\n[OK] Cleaned dataset saved to: {PROCESSED_DATA_PATH}")
print(f"     Shape: {df[save_cols].shape[0]} rows × {len(save_cols)} columns")
print(f"     Columns: {save_cols}")

# --- 9. Missing value summary ---
missing_summary = pd.DataFrame({
    "column": df.columns,
    "missing_count": df.isnull().sum().values,
    "missing_pct": (df.isnull().sum().values / len(df) * 100).round(2)
})
missing_summary.to_csv(os.path.join(TABLES_PATH, "missing_value_summary.csv"), index=False)
print(f"\n[OK] Missing value summary saved to: outputs/tables/missing_value_summary.csv")

# --- 10. Type distribution table ---
type_dist = df["type"].value_counts().reset_index()
type_dist.columns = ["mbti_type", "count"]
type_dist["percentage"] = (type_dist["count"] / type_dist["count"].sum() * 100).round(2)
type_dist.to_csv(os.path.join(TABLES_PATH, "type_distribution.csv"), index=False)
print(f"[OK] Type distribution table saved to: outputs/tables/type_distribution.csv")

print("\nDone.")
