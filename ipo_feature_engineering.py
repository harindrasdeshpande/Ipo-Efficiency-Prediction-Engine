import pandas as pd
import numpy as np

# Load pilot IPO dataset
df = pd.read_csv("ipo_master_raw.csv")

print("\n--- RAW DATA ---")
print(df)


# Define efficiency class based on Listing Day Return

def efficiency_label(ret):
    if ret > 0.20:
        return "Underpriced"
    elif ret < -0.10:
        return "Overpriced"
    else:
        return "Fair"

df["Efficiency_Class"] = df["Return_D0"].apply(efficiency_label)

print("\n--- WITH EFFICIENCY CLASS ---")
print(df[["Company", "Return_D0", "Efficiency_Class"]])


print("\n--- CLASS DISTRIBUTION ---")
print(df["Efficiency_Class"].value_counts())


# For now, only D0 is available
df["Target_Return_D0"] = df["Return_D0"]

# Placeholders for future expansion
df["Target_Return_1W"] = np.nan
df["Target_Return_1M"] = np.nan

print("\n--- FINAL FEATURE TABLE (PILOT) ---")
print(df)


# STEP 3: ADD FUNDAMENTAL FEATURES (PILOT VALUES)

fundamentals = {
    "Zomato":         [2190, 0.25, -0.15, -0.20, 0.30, 9375],
    "Paytm":         [3932, 0.30, -0.20, -0.25, 0.10, 18300],
    "Nykaa":         [2442, 0.40,  0.06,  0.25, 0.05, 5352],
    "IRCTC":         [3500, 0.15,  0.30,  0.35, 0.00, 645],
    "LIC":           [600000,0.05, 0.05,  0.08, 0.20, 210000],
    "Policybazaar": [2400, 0.35, -0.10, -0.15, 0.25, 6017],
    "Uber":          [11139,0.20, -0.15, -0.10, 0.50, 8100],
    "Airbnb":        [3378, 0.35,  0.10,  0.20, 0.10, 3500],
    "Snowflake":    [592,  0.70, -0.40, -0.30, 0.20, 3800],
    "Facebook":     [5089, 0.45,  0.30,  0.40, 0.00, 16000],
    "Alibaba":      [70800,0.50,  0.25,  0.35, 0.10, 25000],
    "Coinbase":     [1300, 0.60,  0.20,  0.30, 0.00, 8600],
}

fund_df = pd.DataFrame.from_dict(
    fundamentals,
    orient="index",
    columns=["Revenue", "Revenue_Growth", "Profit_Margin", "ROE", "Debt_Equity", "Issue_Size"]
).reset_index().rename(columns={"index": "Company"})

df = df.merge(fund_df, on="Company", how="left")

print("\n--- WITH FUNDAMENTAL FEATURES ---")
print(df)

print("\n--- FEATURE SUMMARY ---")
print(df.describe())


# STEP 4: ADD MARKET CONTEXT FEATURES (PILOT VALUES)

market_context = {
    "Zomato":         [0.10, 0.18, 0.12],
    "Paytm":         [0.08, 0.20, 0.10],
    "Nykaa":         [0.12, 0.16, 0.15],
    "IRCTC":         [0.05, 0.14, 0.08],
    "LIC":           [-0.02,0.22, -0.05],
    "Policybazaar": [0.07, 0.19, 0.09],
    "Uber":          [0.06, 0.17, 0.05],
    "Airbnb":        [0.15, 0.25, 0.20],
    "Snowflake":    [0.18, 0.28, 0.25],
    "Facebook":     [0.10, 0.15, 0.12],
    "Alibaba":      [0.09, 0.16, 0.11],
    "Coinbase":     [0.20, 0.30, 0.35],
}

market_df = pd.DataFrame.from_dict(
    market_context,
    orient="index",
    columns=["Market_Return_3M", "Market_Volatility_3M", "Sector_Momentum"]
).reset_index().rename(columns={"index": "Company"})

df = df.merge(market_df, on="Company", how="left")

print("\n--- WITH MARKET CONTEXT FEATURES ---")
print(df)

print("\n--- FINAL DATASET SUMMARY ---")
print(df.info())
print(df.describe())


# STEP 5: FINAL FEATURE SELECTION

# Define target variables
y_class = df["Efficiency_Class"]
y_reg_d0 = df["Target_Return_D0"]

# Drop leakage and non-feature columns
feature_cols = [
    "Issue_Price",
    "Revenue",
    "Revenue_Growth",
    "Profit_Margin",
    "ROE",
    "Debt_Equity",
    "Issue_Size",
    "Market_Return_3M",
    "Market_Volatility_3M",
    "Sector_Momentum",
]

X = df[feature_cols]

print("\n--- FINAL FEATURE MATRIX ---")
print(X.head())

print("\n--- TARGET (CLASSIFICATION) ---")
print(y_class.value_counts())

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_class_encoded = le.fit_transform(y_class)

print("\n--- CLASS LABEL MAPPING ---")
for cls, code in zip(le.classes_, range(len(le.classes_))):
    print(cls, "->", code)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train_class, y_test_class = train_test_split(
    X, y_class_encoded, test_size=0.3, random_state=42 
)

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X, y_reg_d0, test_size=0.3, random_state=42
)

print("\nTrain size:", X_train.shape)
print("Test size:", X_test.shape)


# STEP 6: BASELINE CLASSIFICATION MODEL

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

clf = LogisticRegression(
    solver="lbfgs",
    max_iter=1000
)


clf.fit(X_train, y_train_class)

y_pred_class = clf.predict(X_test)

print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(y_test_class, y_pred_class))

print("\n--- CONFUSION MATRIX ---")
print(confusion_matrix(y_test_class, y_pred_class))

coef_df = pd.DataFrame(
    clf.coef_,
    columns=feature_cols,
    index=le.classes_
)

print("\n--- MODEL COEFFICIENTS (BY CLASS) ---")
print(coef_df)


# STEP 7: REGRESSION MODEL FOR LISTING RETURN

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

reg = LinearRegression()
reg.fit(X_train_reg, y_train_reg)

y_pred_reg = reg.predict(X_test_reg)

print("\n--- REGRESSION PERFORMANCE ---")
print("MSE:", round(mean_squared_error(y_test_reg, y_pred_reg), 5))
print("R2:", round(r2_score(y_test_reg, y_pred_reg), 3))

reg_results = pd.DataFrame({
    "Actual_Return_D0": y_test_reg.values,
    "Predicted_Return_D0": y_pred_reg
})

print("\n--- ACTUAL VS PREDICTED RETURNS ---")
print(reg_results)

reg_coef = pd.Series(reg.coef_, index=feature_cols).sort_values(ascending=False)

print("\n--- REGRESSION COEFFICIENTS (RISK DRIVERS) ---")
print(reg_coef)



"""
FINAL RESULTS SUMMARY

This project builds a pilot IPO efficiency prediction engine using a mixed dataset of Indian and US IPOs.

Models Implemented:
1. Multiclass Logistic Regression:
   - Task: Classify IPOs as Underpriced, Fair, or Overpriced
   - Features: Pre-IPO fundamentals + market regime indicators
   - Dataset size: 12 IPOs (pilot sample)
   - Observations:
       - Model demonstrates ability to separate underpriced vs fair IPOs
       - Performance is unstable due to extreme small-sample and class imbalance

2. Linear Regression:
   - Task: Predict listing-day return percentage
   - Observations:
       - Coefficients show economically meaningful signs
       - Predictive accuracy is limited due to small sample size

Key Finding:
IPO efficiency is driven jointly by:
- Firm quality (profitability, ROE, growth)
- Market timing (pre-IPO market return and volatility)
- Issue size and leverage

This is a pilot research model designed to scale to 60–70 IPOs for production-grade performance.
"""


"""
RESEARCH LIMITATIONS

1. Extremely small sample size (n = 12) limits statistical power
2. Severe class imbalance (only one overpriced IPO)
3. Fundamental and market features are manually approximated in pilot phase
4. No cross-validation used due to sample constraints
5. Transactional and allocation constraints are not modeled

FUTURE WORK

1. Expand dataset to 60–70 IPOs across India and US
2. Automate data collection from prospectuses and financial APIs
3. Add 1-week and 1-month post-listing targets
4. Implement tree-based models (Random Forest, XGBoost)
5. Perform cross-validation and out-of-sample robustness testing
"""
