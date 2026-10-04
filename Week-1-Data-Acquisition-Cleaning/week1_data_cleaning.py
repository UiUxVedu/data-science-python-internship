import pandas as pd
import numpy as np

columns = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week",
    "native-country", "income"
]

df = pd.read_csv("adult.csv", names=columns, skipinitialspace=True)

# Standardize categorical text
categorical_columns = df.select_dtypes(include="object").columns
for col in categorical_columns:
    df[col] = df[col].str.strip()

# Convert ? into real missing values
df = df.replace("?", np.nan)

# Check missing values
print(df.isna().sum())

# Remove exact duplicates
print("Duplicates before:", df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)
print("Duplicates after:", df.duplicated().sum())

# Preserve rows by using an explicit Unknown category
categorical_missing = ["workclass", "occupation", "native-country"]
for col in categorical_missing:
    df[col] = df[col].fillna("Unknown")

# Domain-range checks
range_checks = {
    "age": (17, 90),
    "education-num": (1, 16),
    "hours-per-week": (1, 99)
}

for col, (low, high) in range_checks.items():
    invalid = df[(df[col] < low) | (df[col] > high)]
    print(col, "invalid rows:", len(invalid))

# IQR outlier review
numeric_columns = [
    "age", "fnlwgt", "education-num",
    "capital-gain", "capital-loss", "hours-per-week"
]

for col in numeric_columns:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(col, "IQR outliers:", len(outliers))

# Target encoding
df["income"] = df["income"].map({"<=50K": 0, ">50K": 1})

# Save cleaned data
df.to_csv("adult_cleaned.csv", index=False)

print("Final shape:", df.shape)
print(df.head())
