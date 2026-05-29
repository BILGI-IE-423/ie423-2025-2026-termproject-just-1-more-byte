"""
01_load_data.py
---------------
IE 423 — Term Project
Script 1: Load and inspect the raw MBTI dataset.

This script:
- Loads mbti_1.csv from data/raw/
- Prints basic dataset information (shape, columns, types)
- Shows a sample of the data
- Checks for missing values
"""

import os
import sys

import pandas as pd

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.paths import RAW_DATA

# --- 1. Load dataset ---
print("Loading dataset...")

if not os.path.exists(RAW_DATA):
    print(f"\n[ERROR] Raw data file not found: {RAW_DATA}")
    print("        Download mbti_1.csv from Kaggle and place it in data/raw/.")
    sys.exit(1)

df = pd.read_csv(RAW_DATA)
print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")

# --- 2. Basic information ---
print("\nBasic dataset info:")

print(f"\nShape             : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Column names      : {list(df.columns)}")

print("\nData types:")
print(df.dtypes)

# --- 3. Missing values ---
print("\nMissing value check:")

missing = df.isnull().sum()
print(f"\nMissing values per column:\n{missing}")
if missing.sum() == 0:
    print("\n[OK] No missing values found.")
else:
    print(f"\n[WARNING] Total missing values: {missing.sum()}")

# --- 4. Personality type distribution ---
print("\nPersonality type counts:")

type_counts = df["type"].value_counts()
print(f"\nNumber of unique MBTI types: {df['type'].nunique()}")
print("\nCount per type:")
print(type_counts.to_string())

# --- 5. Sample rows ---
print("\nFirst 3 rows:")

for i, row in df.head(3).iterrows():
    print(f"\n[Row {i}] Type: {row['type']}")
    preview = row["posts"][:200].replace("|||", " | ")
    print(f"  Posts (preview): {preview}...")

print("\nDone.")
