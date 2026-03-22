# Hospital Readmission Analytics & Prediction (Lakehouse Project)

## Overview
This project analyzes hospital readmission performance using CMS (Centers for Medicare & Medicaid Services) data.  
It combines data engineering, exploratory analysis, and machine learning within a Databricks Lakehouse architecture to identify patterns and build predictive models for hospital readmission outcomes.  
The workflow follows a Bronze → Silver → Gold pipeline, enabling scalable and production-style data processing.

## Objective
Explore the Gold ML feature table and generate project-ready insights and predictive models to understand hospital readmission performance.
- End-to-end Lakehouse pipeline (Bronze → Silver → Gold)
- Advanced EDA with geographic visualization
- Clustering + PCA for segmentation
- Regression & Classification modeling
- Feature importance with leakage awareness
- Production-style project structure

## Architecture
- **Bronze Layer**: Raw CMS datasets ingested  
- **Silver Layer**: Cleaned and joined datasets  
- **Gold Layer**: Feature-engineered tables for analytics and ML  

## Exploratory Data Analysis
Key analyses performed:
- Distribution of readmission metrics  
- Hospital rating and performance comparisons  
- Geographic analysis using choropleth maps  
- State-level aggregation of readmission ratios  
- Bubble visualizations incorporating hospital counts  

## Clustering & PCA
- Applied K-Means clustering to group hospitals  
- Used PCA for dimensionality reduction and variance analysis  

Identified distinct hospital profiles:
- Large metropolitan hospitals  
- Community hospitals  
- High-cost hospitals  
- Underperforming hospitals  
- High-quality hospitals  

## Machine Learning Models

### Regression
Models:
- Linear Regression  
- Random Forest Regressor  

Target:
- `excess_readmission_ratio`

### Classification
Models:
- Logistic Regression  
- Random Forest Classifier  

Target:
- `high_readmission_flag`

## Model Performance Summary

### Regression Results

| Model             | MAE    | RMSE   | R²    |
|------------------|--------|--------|-------|
| Linear Regression | 0.019 | 0.037 | 0.786 |
| Random Forest     | 0.001 | 0.005 | 0.995 |

Random Forest significantly outperformed linear regression.

### Classification Results

| Model              | Accuracy | Precision | Recall | F1    | ROC-AUC |
|-------------------|----------|-----------|--------|-------|--------|
| Logistic Regression | 0.998  | 0.999     | 0.996 | 0.998 | 1.00   |
| Random Forest       | 1.000  | 1.000     | 1.000 | 1.000 | 1.00   |

Both models performed extremely well, with Random Forest slightly outperforming.

## Feature Importance Insights
Top predictors:
- Readmission gap  
- Expected readmission rate  
- Predicted readmission rate  

These variables are derived from CMS methodologies and are closely related to the target variable, which may introduce information overlap rather than representing fully independent drivers.

## Key Findings
- Readmission-related metrics dominate predictive performance  
- Random Forest models outperform linear models  
- Geographic and rural indicators have relatively low predictive importance  
- Model performance is extremely high due to strong correlation between features and target  

## Modeling Consideration
Some high-importance features are derived from CMS calculations and are closely tied to the target variable.  
This introduces information overlap (potential leakage), meaning the model is highly predictive but may not fully capture independent causal drivers.

## Future Work
- Remove CMS-derived variables to reduce leakage  
- Incorporate demographic and socioeconomic data (SDOH)  
- Explore time-series modeling for trend analysis  
- Integrate MLflow tracking in a production-enabled environment  

## Author
**Mehmet A. Comert**  
M.S. Data Science