# IPO Listing Efficiency Prediction Engine

## Overview
This project builds a pilot machine learning system to predict IPO listing efficiency using pre-IPO company fundamentals and market regime indicators.

The objective is to classify IPOs as:
- Underpriced
- Fairly Priced
- Overpriced

and simultaneously predict the expected listing-day return percentage.

The project is designed as a scalable research framework that can expand to 60–70 IPOs for production-grade modeling.

---

## Problem Framing

Two supervised learning tasks are implemented:

1. Classification Task  
Predict IPO efficiency class:
- Underpriced (Return > +20%)
- Fair (−10% to +20%)
- Overpriced (Return < −10%)

2. Regression Task  
Predict continuous listing-day return percentage.

All predictors are strictly limited to information available before the IPO date to avoid lookahead bias.

---

## Dataset

Pilot dataset:
- 12 IPOs (India + US)
- Features include:
  - Issue price
  - Revenue, growth, profitability
  - ROE, leverage
  - Issue size
  - Pre-IPO market return and volatility
  - Sector momentum

Targets:
- Efficiency_Class (classification)
- Target_Return_D0 (regression)

---

## Models Implemented

1. Multiclass Logistic Regression  
- Task: IPO efficiency classification  
- Output: Class probabilities for Underpriced / Fair / Overpriced  

2. Linear Regression  
- Task: Predict listing-day return  
- Output: Expected return percentage  

Both models use only pre-IPO features.

---

## Key Results (Pilot Phase)

- The classification model demonstrates separation between underpriced and fairly priced IPOs.
- Regression coefficients show economically meaningful relationships between:
  - Profitability and positive listing returns  
  - Market regime and IPO performance  

Due to the very small pilot sample, predictive accuracy is unstable and intended for methodological demonstration.

---

## Research Limitations

- Extremely small sample size (n = 12)
- Severe class imbalance
- Manually approximated fundamentals in pilot phase
- No cross-validation due to sample constraints

---

## Future Work

- Expand dataset to 60–70 IPOs  
- Automate financial data ingestion  
- Add 1-week and 1-month performance targets  
- Implement Random Forest and Gradient Boosting  
- Perform cross-validation and regime-specific evaluation  

---

## Tools & Technologies

- Python: Data engineering, modeling
- Scikit-learn: Classification and regression
- Pandas / NumPy: Feature engineering
- Planned: Go-based deployment engine

---

## Key Learning

This project demonstrates how IPO efficiency can be modeled as a joint function of firm fundamentals and market timing, with strict control of information leakage and research-grade modeling discipline.
