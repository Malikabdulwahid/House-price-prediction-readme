
# HOUSE PRICE PREDICTION - COMPLETE ML PIPELINE

# STEP 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# ─────────────────────────────────────────────────
# STEP 2: Load Dataset
# ─────────────────────────────────────────────────
import pandas as pd

df = pd.read_csv(r"C:\Users\hp\Desktop\Fitness\house_price_dataset.csv")

print(df.head())
print("=" * 60)
print("STEP 2: DATASET OVERVIEW")
print("=" * 60)
print(f"Shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print("\nFirst 5 rows:")
print(df.head())
print("\nData Types:")
print(df.dtypes)

# ─────────────────────────────────────────────────
# STEP 3: Exploratory Data Analysis (EDA)
# ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nCategorical Value Counts:")
for col in ['location', 'condition', 'garage', 'furnishing']:
    print(f"\n{col}:\n{df[col].value_counts(dropna=False)}")

# Plot 1: Target Distribution
plt.figure(figsize=(8, 4))
sns.histplot(df['price'], kde=True, color='steelblue')
plt.title('House Price Distribution')
plt.xlabel('Price')
plt.tight_layout()
plt.savefig('price_distribution.png')
plt.show()

# Plot 2: Correlation Heatmap (numeric only)
plt.figure(figsize=(10, 8))
numeric_df = df.select_dtypes(include=np.number).drop(columns=['id'])
sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.show()

# Plot 3: Boxplots for categorical vs price
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
for ax, col in zip(axes, ['location', 'condition', 'garage', 'furnishing']):
    temp = df[[col, 'price']].dropna()
    sns.boxplot(x=col, y='price', data=temp, ax=ax)
    ax.set_title(f'Price by {col}')
    ax.tick_params(axis='x', rotation=30)
plt.tight_layout()
plt.savefig('categorical_boxplots.png')
plt.show()

# Plot 4: Scatter - Area vs Price
plt.figure(figsize=(7, 5))
plt.scatter(df['area'], df['price'], alpha=0.4, color='coral')
plt.xlabel('Area')
plt.ylabel('Price')
plt.title('Area vs Price')
plt.tight_layout()
plt.savefig('area_vs_price.png')
plt.show()

# ─────────────────────────────────────────────────
# STEP 4: Data Preprocessing
# ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: DATA PREPROCESSING")
print("=" * 60)

df_clean = df.copy()

# Drop ID column
df_clean.drop(columns=['id'], inplace=True)

# --- Fix Inconsistent Categories ---
df_clean['location'] = df_clean['location'].str.strip().str.capitalize()
df_clean['furnishing'] = df_clean['furnishing'].replace({'Semi': 'Semifurnished'})

print("After fix - location unique:", df_clean['location'].unique())
print("After fix - furnishing unique:", df_clean['furnishing'].unique())

# --- Handle Missing Values ---
# Numerical: fill with median
num_cols_with_na = ['area', 'age', 'income']
for col in num_cols_with_na:
    median_val = df_clean[col].median()
    df_clean[col].fillna(median_val, inplace=True)
    print(f"Filled '{col}' NaN with median: {median_val:.2f}")

# Categorical: fill with mode
cat_cols_with_na = ['location', 'garage']
for col in cat_cols_with_na:
    mode_val = df_clean[col].mode()[0]
    df_clean[col].fillna(mode_val, inplace=True)
    print(f"Filled '{col}' NaN with mode: {mode_val}")

print(f"\nMissing values after cleaning:\n{df_clean.isnull().sum()}")

# --- Encode Categorical Variables ---
le = LabelEncoder()
cat_cols = ['location', 'condition', 'garage', 'furnishing']
for col in cat_cols:
    df_clean[col] = le.fit_transform(df_clean[col].astype(str))
    print(f"Encoded '{col}'")

print("\nData after encoding (head):")
print(df_clean.head())

# ─────────────────────────────────────────────────
# STEP 5: Feature Engineering
# ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: FEATURE ENGINEERING")
print("=" * 60)

# New feature: price per sq ft proxy (area-income ratio)
df_clean['area_income_ratio'] = df_clean['area'] / (df_clean['income'] + 1)

# New feature: total rooms
df_clean['total_rooms'] = df_clean['bedrooms'] + df_clean['bathrooms']

print("New features added: 'area_income_ratio', 'total_rooms'")
print(df_clean[['area_income_ratio', 'total_rooms', 'price']].head())

# ─────────────────────────────────────────────────
# STEP 6: Train-Test Split
# ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: TRAIN-TEST SPLIT")
print("=" * 60)

X = df_clean.drop(columns=['price'])
y = df_clean['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set: {X_train.shape}")
print(f"Testing set:  {X_test.shape}")

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ─────────────────────────────────────────────────
# STEP 7: Model Training & Evaluation
# ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: MODEL TRAINING & EVALUATION")
print("=" * 60)

models = {
    'Linear Regression'       : LinearRegression(),
    'Ridge Regression'        : Ridge(alpha=10),
    'Lasso Regression'        : Lasso(alpha=10),
    'Decision Tree'           : DecisionTreeRegressor(random_state=42),
    'Random Forest'           : RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting'       : GradientBoostingRegressor(n_estimators=100, random_state=42),
}

results = {}

for name, model in models.items():
    # Use scaled data for linear models, unscaled for tree-based
    if 'Regression' in name:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)

    results[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
    print(f"\n{name}")
    print(f"  MAE  : {mae:,.2f}")
    print(f"  RMSE : {rmse:,.2f}")
    print(f"  R²   : {r2:.4f}")

# ─────────────────────────────────────────────────
# STEP 8: Model Comparison Plot
# ─────────────────────────────────────────────────
results_df = pd.DataFrame(results).T.sort_values('R2', ascending=False)
print("\n" + "=" * 60)
print("STEP 8: MODEL COMPARISON")
print("=" * 60)
print(results_df)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
metrics = ['MAE', 'RMSE', 'R2']
colors  = ['#e74c3c', '#e67e22', '#2ecc71']
for ax, metric, color in zip(axes, metrics, colors):
    results_df[metric].sort_values().plot(kind='barh', ax=ax, color=color)
    ax.set_title(f'Model {metric}')
    ax.set_xlabel(metric)
plt.tight_layout()
plt.savefig('model_comparison.png')
plt.show()

# ─────────────────────────────────────────────────
# STEP 9: Best Model - Feature Importance
# ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 9: BEST MODEL - RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 60)

best_model = RandomForestRegressor(n_estimators=100, random_state=42)
best_model.fit(X_train, y_train)
y_pred_best = best_model.predict(X_test)

feature_importance = pd.Series(best_model.feature_importances_, index=X.columns)
feature_importance = feature_importance.sort_values(ascending=False)

print(feature_importance)

plt.figure(figsize=(10, 6))
feature_importance.plot(kind='bar', color='steelblue')
plt.title('Random Forest - Feature Importance')
plt.ylabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()

# ─────────────────────────────────────────────────
# STEP 10: Actual vs Predicted Plot
# ─────────────────────────────────────────────────
plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred_best, alpha=0.4, color='purple')
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Random Forest: Actual vs Predicted')
plt.tight_layout()
plt.savefig('actual_vs_predicted.png')
plt.show()

# ─────────────────────────────────────────────────
# STEP 11: Cross-Validation
# ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 11: CROSS-VALIDATION (5-Fold) - Random Forest")
print("=" * 60)

cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='r2')
print(f"CV R² Scores : {cv_scores}")
print(f"Mean R²      : {cv_scores.mean():.4f}")
print(f"Std Dev      : {cv_scores.std():.4f}")

# ─────────────────────────────────────────────────
# STEP 12: Save Model & Scaler
# ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 12: SAVING MODEL & SCALER")
print("=" * 60)

joblib.dump(best_model, 'house_price_model.pkl')
joblib.dump(scaler,     'scaler.pkl')
joblib.dump(list(X.columns), 'feature_names.pkl')

print("Saved: house_price_model.pkl")
print("Saved: scaler.pkl")
print("Saved: feature_names.pkl")
print("\n✅ ML Pipeline Complete!")
