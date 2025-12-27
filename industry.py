# ================================
# 1. IMPORT LIBRARIES
# ================================
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set(style="whitegrid")


# ================================
# 2. LOAD DATASET
# ================================
df = pd.read_csv("Housing.csv")

print("Initial Shape:", df.shape)
print(df.head())


# ================================
# 3. BASIC INSPECTION
# ================================
print("\nINFO:")
df.info()

print("\nDESCRIBE:")
print(df.describe())


# ================================
# 4. CHECK & HANDLE MISSING VALUES
# ================================
print("\nMissing Values:")
print(df.isnull().sum())

# Numerical → Median
num_cols = df.select_dtypes(include=np.number).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Categorical → Mode
cat_cols = df.select_dtypes(include="object").columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])


# ================================
# 5. ENCODE CATEGORICAL VARIABLES
# ================================
df = pd.get_dummies(df, drop_first=True)

print("\nShape after encoding:", df.shape)


# ================================
# 6. EXPLORATORY DATA ANALYSIS
# ================================

# Target distribution
plt.figure(figsize=(6,4))
sns.histplot(df["price"], kde=True)
plt.title("House Price Distribution")
plt.show()

# Correlation heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap")
plt.show()

# Feature vs target plots
important_features = ["area", "bedrooms", "bathrooms"]

for feature in important_features:
    plt.figure(figsize=(5,4))
    sns.scatterplot(x=df[feature], y=df["price"])
    plt.title(f"{feature} vs Price")
    plt.show()


# ================================
# 7. FEATURE–TARGET SPLIT
# ================================
X = df.drop("price", axis=1)
y = df["price"]


# ================================
# 8. TRAIN–TEST SPLIT
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)


# ================================
# 9. MODEL TRAINING
# ================================
model = LinearRegression()
model.fit(X_train, y_train)


# ================================
# 10. MODEL PREDICTION
# ================================
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)


# ================================
# 11. MODEL EVALUATION
# ================================
mae = mean_absolute_error(y_test, y_test_pred)
mse = mean_squared_error(y_test, y_test_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_test_pred)

print("\nMODEL PERFORMANCE")
print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R2  :", r2)


# ================================
# 12. COEFFICIENT INTERPRETATION
# ================================
coeff_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
}).sort_values(by="Coefficient", ascending=False)

print("\nFeature Coefficients:")
print(coeff_df)


# ================================
# 13. RESIDUAL ANALYSIS
# ================================
residuals = y_test - y_test_pred

# Residuals vs Predicted
plt.figure(figsize=(6,4))
sns.scatterplot(x=y_test_pred, y=residuals)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted Price")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted")
plt.show()

# Residual distribution
plt.figure(figsize=(6,4))
sns.histplot(residuals, kde=True)
plt.title("Residual Distribution")
plt.show()


# ================================
# 14. FINAL SUMMARY
# ================================
print("\nFINAL SUMMARY")
print("Linear Regression model trained successfully.")
print("Residuals are approximately centered around zero.")
print("RMSE gives average prediction error in price units.")
print("Model is suitable as a baseline regression model.")
