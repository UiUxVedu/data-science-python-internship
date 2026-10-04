import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("adult_cleaned.csv")

print(df.shape)
print(df.info())
print(df.describe(include="all").T)

# Target distribution
print(df["income"].value_counts())
print(df["income"].value_counts(normalize=True) * 100)

sns.countplot(data=df, x="income")
plt.title("Income Category Distribution")
plt.xlabel("Income category")
plt.ylabel("Number of records")
plt.tight_layout()
plt.show()

# Numerical distributions
numeric_cols = [
    "age", "fnlwgt", "education-num",
    "capital-gain", "capital-loss", "hours-per-week"
]

df[numeric_cols].hist(figsize=(12, 8), bins=25)
plt.suptitle("Distributions of Numerical Variables")
plt.tight_layout()
plt.show()

# Categorical analysis
for col in ["workclass", "education", "marital-status", "occupation"]:
    print(df[col].value_counts().head(10))

# Education vs income
education_income = (
    pd.crosstab(df["education"], df["income"], normalize="index") * 100
)

education_income.plot(kind="bar", figsize=(11, 5))
plt.title("Income Distribution Within Each Education Level")
plt.xlabel("Education level")
plt.ylabel("Percentage")
plt.legend(title="Income")
plt.xticks(rotation=55, ha="right")
plt.tight_layout()
plt.show()

# Workclass vs income
workclass_income = (
    pd.crosstab(df["workclass"], df["income"], normalize="index") * 100
)

workclass_income.plot(kind="bar", figsize=(10, 5))
plt.title("Income Distribution Within Each Workclass")
plt.xlabel("Workclass")
plt.ylabel("Percentage")
plt.legend(title="Income")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.show()

# Hours groups
df["hours_group"] = pd.cut(
    df["hours-per-week"],
    bins=[0, 29, 39, 40, 49, 59, 100],
    labels=["<30", "30-39", "40", "41-49", "50-59", "60+"]
)

hours_income = (
    pd.crosstab(df["hours_group"], df["income"], normalize="index") * 100
)

hours_income.plot(kind="bar", figsize=(9, 5))
plt.title("Income Distribution by Weekly Working Hours")
plt.xlabel("Hours per week")
plt.ylabel("Percentage")
plt.legend(title="Income")
plt.tight_layout()
plt.show()

# Age vs income
sns.boxplot(data=df, x="income", y="age")
plt.title("Age Distribution by Income Category")
plt.xlabel("Income category")
plt.ylabel("Age")
plt.tight_layout()
plt.show()

# Correlation
corr_cols = [
    "age", "education-num", "capital-gain",
    "capital-loss", "hours-per-week"
]

corr = df[corr_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Matrix of Numerical Variables")
plt.tight_layout()
plt.show()

# Multivariate view
sns.boxplot(data=df, x="income", y="hours-per-week", hue="sex")
plt.title("Weekly Working Hours by Income and Sex")
plt.xlabel("Income category")
plt.ylabel("Hours per week")
plt.legend(title="Sex")
plt.tight_layout()
plt.show()
