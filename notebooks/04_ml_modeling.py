# Databricks notebook source
# MAGIC %md
# MAGIC # Hospital Readmission Prediction Models

# COMMAND ----------

# MAGIC %md
# MAGIC # 1 Objective
# MAGIC The goal is to evaluate how well operational metrics, hospital quality indicators, spending measures, and geographic context can predict hospital readmission performance.
# MAGIC
# MAGIC This notebook builds predictive models using the Gold ML feature table prepared in the lakehouse pipeline.
# MAGIC
# MAGIC Two modeling tasks are explored:
# MAGIC
# MAGIC 1. **Regression:** Predict hospital excess readmission ratio.  
# MAGIC 2. **Classification:** Predict whether a hospital has a high readmission burden.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Feature Selection
# MAGIC
# MAGIC The modeling dataset is constructed from the **Gold ML feature table** produced in the lakehouse pipeline.
# MAGIC
# MAGIC Features were selected based on the exploratory data analysis and their potential relevance to hospital readmission performance. These variables represent hospital operational scale, quality performance, spending behavior, and geographic context — all of which may influence hospital readmission outcomes.
# MAGIC
# MAGIC The selected features fall into several categories:
# MAGIC
# MAGIC ### Hospital Utilization Metrics
# MAGIC These variables capture hospital patient volume and operational activity. Higher patient volume may influence readmission patterns and hospital workload.
# MAGIC
# MAGIC - number_of_discharges  
# MAGIC - number_of_readmissions  
# MAGIC - total_unplanned_patients  
# MAGIC - total_unplanned_patients_returned  
# MAGIC - total_unplanned_denominator  
# MAGIC
# MAGIC ### Readmission Performance Indicators
# MAGIC CMS provides predicted and expected readmission metrics which help estimate how hospitals perform relative to national benchmarks.
# MAGIC
# MAGIC - predicted_readmission_rate  
# MAGIC - expected_readmission_rate  
# MAGIC - unplanned_return_rate  
# MAGIC
# MAGIC ### Hospital Quality Indicators
# MAGIC These variables represent hospital quality signals and patient care outcomes.
# MAGIC
# MAGIC - hospital_overall_rating  
# MAGIC - birthing_friendly_designation  
# MAGIC - emergency_services  
# MAGIC
# MAGIC ### Cost and Resource Utilization
# MAGIC Spending metrics provide insight into hospital resource use and operational efficiency.
# MAGIC
# MAGIC - mspb_score  
# MAGIC - avg_unplanned_score  
# MAGIC
# MAGIC ### Geographic Context
# MAGIC Geographic indicators help capture differences in hospital location and access to care.
# MAGIC
# MAGIC - ruca_code  
# MAGIC - secondary_ruca_code  
# MAGIC
# MAGIC Identifier fields and purely descriptive attributes such as **facility_id**, **facility_name**, **state**, and **zip_code** were excluded from modeling because they do not provide predictive signal for hospital performance.
# MAGIC
# MAGIC These selected features will be used to train both **regression** and **classification** models in the following sections.

# COMMAND ----------

# MAGIC %md
# MAGIC # 2 Load Gold ML Feature Table
# MAGIC The modeling dataset is sourced from the **Gold ML feature table** created in the lakehouse pipeline.  
# MAGIC This table contains cleaned and engineered features derived from CMS hospital data and serves as the input dataset for machine learning models.
# MAGIC
# MAGIC The dataset includes hospital operational metrics, quality indicators, spending measures, and geographic attributes that may influence hospital readmission outcomes.

# COMMAND ----------

GOLD = "/Volumes/workspace/cms_hospital/gold"
gold_ml = spark.read.format("delta").load(f"{GOLD}/ml_features")

display(gold_ml)

# COMMAND ----------

gold_ml.columns

# COMMAND ----------

print("Total records:", gold_ml.count())
print("Total columns", len(gold_ml.columns), "\n", "-"* 20)

gold_ml.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC # 3 Feature Selection
# MAGIC The modeling dataset is constructed from the **Gold ML feature table** produced in the lakehouse pipeline.
# MAGIC
# MAGIC Features were selected based on the exploratory data analysis and their potential relevance to hospital readmission performance. These variables represent hospital operational scale, quality performance, spending behavior, and geographic context — all of which may influence hospital readmission outcomes.
# MAGIC
# MAGIC Identifier fields and purely descriptive attributes such as **facility_id**, **facility_name**, **state**, and **zip_code** were excluded from modeling because they do not provide predictive signal for hospital performance.
# MAGIC
# MAGIC The selected features fall into several categories including hospital utilization metrics, readmission performance indicators, quality indicators, cost metrics, and geographic context variables.
# MAGIC
# MAGIC Note: The ML feature table excludes raw discharge and readmission counts because derived utilization metrics and readmission indicators already capture the relevant information for modeling.

# COMMAND ----------

# Select modeling features

feature_cols = [
    "hospital_overall_rating",
    "mspb_score",
    "avg_unplanned_score",
    "total_unplanned_denominator",
    "total_unplanned_patients",
    "total_unplanned_patients_returned",
    "unplanned_return_rate",
    "ruca_code",
    "secondary_ruca_code",
    "is_emergency_services",
    "has_hospital_rating",
    "high_quality_hospital",
    "high_patient_volume",
    "is_rural",
    "predicted_readmission_rate",
    "expected_readmission_rate",
    "readmission_gap"
]

# Target columns for regression and classification
target_reg = "excess_readmission_ratio"
target_clf = "high_readmission_flag"

# COMMAND ----------

# MAGIC %md
# MAGIC # 4 Modeling Dataset Preparation
# MAGIC This section prepares the final modeling dataset using the selected feature columns and target variables from the Gold ML feature table.
# MAGIC
# MAGIC The preparation steps include:
# MAGIC
# MAGIC - selecting model input features and target variables
# MAGIC - removing rows with missing values in the selected modeling fields
# MAGIC - converting the Spark DataFrame to Pandas for scikit-learn modeling
# MAGIC - reviewing the final dataset shape and previewing the input data
# MAGIC
# MAGIC Feature scaling will be applied later where appropriate. Linear and logistic regression models require scaled inputs, while tree-based models such as Random Forest do not.

# COMMAND ----------

# Model column selections
model_cols = feature_cols + [ target_reg, target_clf]

# Drop nan values
ml_df = gold_ml.select(*model_cols).dropna()

display(ml_df.limit(5))

# COMMAND ----------

# Validate dataframes number of records
print("gold_ml records: ", gold_ml.count())
print("ml_df records: ", ml_df.count())

# COMMAND ----------

# Record counts for comparison with 
gold_count = gold_ml.count()
model_count = ml_df.count()

print("Gold ML rows:", gold_count)
print("Model dataset rows:", model_count)
print("Rows removed:", gold_count - model_count)
print("Percent removed:", round((gold_count - model_count) / gold_count * 100, 2), "%")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 4.0.1 Modeling dataset size output note:
# MAGIC After selecting the modeling features and removing rows with missing values in the selected fields, the final modeling dataset contains **10,916 records**.
# MAGIC
# MAGIC This reduction occurs because several hospital quality and utilization metrics are not available for all facilities. Removing rows with missing values ensures that the machine learning models are trained on complete feature vectors.

# COMMAND ----------

# Convert spark dataframe to pandas dataframe
ml_pd = ml_df.toPandas()

ml_pd.head()

# COMMAND ----------

# Null check
ml_pd.isnull().sum()

# COMMAND ----------

# Verify target column value counts
print(ml_pd[target_clf].value_counts())

# COMMAND ----------

# Check regression target column stats
print(ml_pd[target_reg].describe())

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 4.0.2 Modeling dataset target regression output note:
# MAGIC The Excess Readmission Ratio (ERR) is calculated by the Centers for Medicare & Medicaid Services (CMS) as part of the Hospital Readmissions Reduction Program. The metric compares a hospital's predicted readmissions to the expected readmissions for similar patients nationwide. Values greater than 1 indicate higher-than-expected readmissions, while values below 1 indicate better-than-expected performance.

# COMMAND ----------

# MAGIC %md
# MAGIC # 5 Train/Test Split and Scaling
# MAGIC
# MAGIC The modeling dataset is divided into training and test sets for both regression and classification tasks.
# MAGIC - Scaling is applied only to models that are sensitive to feature magnitudes, such as Linear Regression and Logistic Regression. 
# MAGIC - Tree-based models such as Random Forest are trained on the original unscaled features.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.1 Define X & y - Regression - Classification

# COMMAND ----------

# Define X and y for both regression and classification

X = ml_pd[feature_cols]
y_reg = ml_pd[target_reg]
y_clf = ml_pd[target_clf]


print("Feature matrix shape:", X.shape)
print("Regression target shape:", y_reg.shape)
print("Classification target shape:", y_clf.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.2 Split Regression Data

# COMMAND ----------

from sklearn.model_selection import train_test_split

# Train/test split for regression
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)

print("Regression train shape:", X_train_reg.shape)
print("Regression test shape:", X_test_reg.shape)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.3 Split Classification Data

# COMMAND ----------

# Train/test split for classification
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

print("Classification train shape:", X_train_clf.shape)
print("Classification test shape:", X_test_clf.shape)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.4 Scale Linear/ Logistic Models Only

# COMMAND ----------

from sklearn.preprocessing import StandardScaler

# Scale regression data
scaler_reg = StandardScaler()
X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
X_test_reg_scaled = scaler_reg.transform(X_test_reg)

# Scale classification data
scaler_clf = StandardScaler()
X_train_clf_scaled = scaler_clf.fit_transform(X_train_clf)
X_test_clf_scaled = scaler_clf.transform(X_test_clf)

print("Scaled regression train shape:", X_train_reg_scaled.shape)
print("Scaled classification train shape:", X_train_clf_scaled.shape)



# COMMAND ----------

# MAGIC %md
# MAGIC # 6 Regression Models
# MAGIC
# MAGIC Two regression models are trained to predict **excess readmission ratio**:
# MAGIC
# MAGIC - **Linear Regression**: a baseline interpretable model
# MAGIC - **Random Forest Regressor**: a non-linear ensemble model that can capture more complex relationships
# MAGIC
# MAGIC Model performance is evaluated using:
# MAGIC
# MAGIC - MAE (Mean Absolute Error)
# MAGIC - RMSE (Root Mean Squared Error)
# MAGIC - R² Score

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.1 Linear Regression
# MAGIC
# MAGIC Linear Regression is used as a baseline model to estimate hospital excess readmission ratio from the selected hospital operational, quality, cost, and geographic features.
# MAGIC
# MAGIC Because linear models are sensitive to feature scale, the standardized training and test datasets are used in this section.

# COMMAND ----------

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Linear regression
lr_model = LinearRegression()
lr_model.fit(X_train_reg_scaled, y_train_reg)

# Predict
lr_ypred = lr_model.predict(X_test_reg_scaled)

# Evaluate
lr_mae = mean_absolute_error(y_test_reg, lr_ypred)
lr_mse = mean_squared_error(y_test_reg, lr_ypred)
lr_rmse = mean_squared_error(y_test_reg, lr_ypred, squared=False)
lr_r2 = r2_score(y_test_reg, lr_ypred)

print("Linear Regression Performance")
print("-------------------------------")
print("MAE_lr:", round(lr_mae,3))
print("MSE_lr:", round(lr_mse, 3))
print("RMSE_lr:", round(lr_rmse, 3))
print("R2_lr:", round(lr_r2, 3))



# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.2 Random Forest Regressor
# MAGIC
# MAGIC Random Forest Regressor is used as a non-linear alternative to Linear Regression. Unlike linear models, Random Forest does not require scaled inputs and can capture more complex feature interactions.

# COMMAND ----------

from sklearn.ensemble import RandomForestRegressor

# Train and fit random forest regressor
rf_reg_model = RandomForestRegressor(
    n_estimators=100,
    max_depth = 10,
    random_state= 42,
    n_jobs= -1
)

rf_reg_model.fit(X_train_reg, y_train_reg)

# Predict on test data
rf_reg_ypred = rf_reg_model.predict(X_test_reg)

# Evaluate
rf_reg_mae = mean_absolute_error(y_test_reg, rf_reg_ypred)
rf_reg_rmse = mean_squared_error(y_test_reg, rf_reg_ypred, squared=False)
rf_reg_r2 = r2_score(y_test_reg, rf_reg_ypred)


print("Random Forest Regressor Performance")
print("-------------------------------")
print("MAE_rf:", round(rf_reg_mae, 3))
print("RMSE_rf:", round(rf_reg_rmse, 3))
print("R²_rf:", round(rf_reg_r2, 3))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.3 Regression Model Comparison
# MAGIC
# MAGIC The performance of Linear Regression and Random Forest Regressor is compared using MAE, RMSE, and R². Lower MAE and RMSE indicate better prediction accuracy, while higher R² indicates a better fit to the target variable.

# COMMAND ----------

import pandas as pd
# Create a pandas DataFrame with the results for each model compared
regression_results = pd.DataFrame(
  {
    "Model": ["Linear Regression", "Random Forest Regressor"],
    "MAE": [lr_mae, rf_reg_mae],
    "RMSE": [lr_rmse, rf_reg_rmse],
    "R2": [lr_r2, rf_reg_r2]
  }
)

regression_results

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.3.1 Regression Model Comparison Chart

# COMMAND ----------

import matplotlib.pyplot as plt

regression_results_plot = regression_results.set_index("Model")

regression_results_plot[["MAE", "RMSE", "R2"]].plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Regression Model Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 6.3.1 Regression model comparision output note:
# MAGIC The regression model comparison shows that the **Random Forest Regressor significantly outperforms the Linear Regression model** across all evaluation metrics.
# MAGIC
# MAGIC Linear Regression achieved an R² score of approximately **0.79**, indicating that the model is able to explain a large portion of the variation in hospital excess readmission ratios using the selected hospital operational, quality, and geographic features. However, the Random Forest model produced substantially lower prediction errors, with a **MAE of 0.00136** and **RMSE of 0.00555**, along with a near-perfect **R² score of 0.995**.
# MAGIC
# MAGIC These results suggest that the relationship between hospital characteristics and readmission performance is likely **non-linear and influenced by complex feature interactions**, which are better captured by the ensemble-based Random Forest model than by the linear baseline model.
# MAGIC
# MAGIC Overall, the Random Forest Regressor demonstrates superior predictive performance and is therefore selected as the **best-performing regression model** for predicting hospital excess readmission ratios in this analysis.

# COMMAND ----------

# MAGIC %md
# MAGIC # 7 Classification Models
# MAGIC This section builds classification models to predict whether a hospital belongs to the **high readmission group**.
# MAGIC
# MAGIC Two classification models are evaluated:
# MAGIC - **Logistic Regression**: a baseline interpretable classifier
# MAGIC - **Random Forest Classifier**: a non-linear ensemble model that can capture more complex feature interactions
# MAGIC
# MAGIC Model performance will be evaluated using:
# MAGIC - Accuracy
# MAGIC - Precision
# MAGIC - Recall
# MAGIC - F1-score
# MAGIC - ROC-AUC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7.1 Logistic Regression
# MAGIC Logistic Regression is used as a baseline classification model for predicting high readmission risk. Because logistic regression is sensitive to feature scale, the standardized feature matrix is used in this section.

# COMMAND ----------

# Import necessary libraries
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# Train Logistic Regression model
log_reg_model = LogisticRegression(max_iter=1000, random_state=42)
log_reg_model.fit(X_train_clf_scaled, y_train_clf)

# Predictions
log_reg_pred = log_reg_model.predict(X_test_clf_scaled)
log_reg_prob = log_reg_model.predict_proba(X_test_clf_scaled)[:, 1]

# Metrics
log_reg_accuracy = accuracy_score(y_test_clf, log_reg_pred)
log_reg_precision = precision_score(y_test_clf, log_reg_pred)
log_reg_recall = recall_score(y_test_clf, log_reg_pred)
log_reg_f1 = f1_score(y_test_clf, log_reg_pred)
log_reg_roc_auc = roc_auc_score(y_test_clf, log_reg_prob)

print("Logistic Regression Performance")
print("Accuracy :", round(log_reg_accuracy, 4))
print("Precision:", round(log_reg_precision, 4))
print("Recall   :", round(log_reg_recall, 4))
print("F1 Score :", round(log_reg_f1, 4))
print("ROC-AUC  :", round(log_reg_roc_auc, 4))


# COMMAND ----------

# MAGIC %md
# MAGIC ### 7.1.1 Classification Report

# COMMAND ----------

# Classification Report
print("Classification Report - Logistic Regression")
print(classification_report(y_test_clf, log_reg_pred))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7.1.2 Confusion Matrix

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

log_reg_cm = confusion_matrix(y_test_clf, log_reg_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(log_reg_cm, annot=True, fmt="d")
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # 7.2 Random Forest Classifier
# MAGIC Random Forest Classifier is used as a non-linear alternative to Logistic Regression. Unlike logistic regression, Random Forest does not require scaled inputs and can capture complex interactions among hospital quality, utilization, and geographic features.

# COMMAND ----------

from sklearn.ensemble import RandomForestClassifier

# Train Random Forest Classifier
rf_clf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

rf_clf_model.fit(X_train_clf, y_train_clf)

# Predictions
rf_clf_pred = rf_clf_model.predict(X_test_clf)
rf_clf_prob = rf_clf_model.predict_proba(X_test_clf)[:, 1]

# Metrics
rf_clf_accuracy = accuracy_score(y_test_clf, rf_clf_pred)
rf_clf_precision = precision_score(y_test_clf, rf_clf_pred)
rf_clf_recall = recall_score(y_test_clf, rf_clf_pred)
rf_clf_f1 = f1_score(y_test_clf, rf_clf_pred)
rf_clf_roc_auc = roc_auc_score(y_test_clf, rf_clf_prob)

print("Random Forest Classifier Performance")
print("Accuracy :", round(rf_clf_accuracy, 4))
print("Precision:", round(rf_clf_precision, 4))
print("Recall   :", round(rf_clf_recall, 4))
print("F1 Score :", round(rf_clf_f1, 4))
print("ROC-AUC  :", round(rf_clf_roc_auc, 4))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7.2.1 Classification Report

# COMMAND ----------

print("Classification Report - Random Forest Classifier")
print(classification_report(y_test_clf, rf_clf_pred))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7.2.2 Confusion Matrix

# COMMAND ----------

rf_clf_cm = confusion_matrix(y_test_clf, rf_clf_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(rf_clf_cm, annot=True, fmt="d", cmap="Greens")
plt.title("Random Forest Classifier Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # 7.3 Classification Model Comparison
# MAGIC The performance of Logistic Regression and Random Forest Classifier is compared using accuracy, precision, recall, F1-score, and ROC-AUC.
# MAGIC
# MAGIC Because the classification target is relatively balanced, these metrics provide a reliable comparison of model performance.

# COMMAND ----------

import pandas as pd

# Create a DataFrame for the classification results
classification_results = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest Classifier"],
    "Accuracy": [log_reg_accuracy, rf_clf_accuracy],
    "Precision": [log_reg_precision, rf_clf_precision],
    "Recall": [log_reg_recall, rf_clf_recall],
    "F1 Score": [log_reg_f1, rf_clf_f1],
    "ROC-AUC": [log_reg_roc_auc, rf_clf_roc_auc]
})

classification_results = classification_results.round(3)
classification_results

# COMMAND ----------

classification_results_plot = classification_results.set_index("Model")

classification_results_plot.plot.bar(rot=0, figsize = (8,6))

plt.title("Classification Model Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 7.3.1 Classification model comparision output note:
# MAGIC The classification model comparison shows that both Logistic Regression and Random Forest Classifier perform extremely well in predicting high readmission hospitals.
# MAGIC
# MAGIC Logistic Regression already achieves near-perfect performance, with an accuracy of **99.8%**, F1-score of **0.998**, and ROC-AUC of **1.0**, indicating that the selected features provide very strong predictive signal for identifying high readmission risk.
# MAGIC
# MAGIC The Random Forest Classifier slightly outperforms Logistic Regression, achieving perfect scores across all evaluation metrics (Accuracy, Precision, Recall, F1-score, and ROC-AUC all equal to **1.0**). This suggests that the non-linear model is able to fully capture the relationships between hospital characteristics and readmission risk.
# MAGIC
# MAGIC The minimal difference between the two models indicates that the classification problem is highly separable based on the selected features. This is likely due to the inclusion of CMS-derived readmission metrics and related variables, which are strongly correlated with the target variable.
# MAGIC
# MAGIC Overall, while both models perform exceptionally well, the Random Forest Classifier is selected as the best-performing model due to its perfect predictive performance.

# COMMAND ----------

# MAGIC %md
# MAGIC # 8 Feature Importance
# MAGIC To better understand which hospital characteristics drive readmission performance, feature importance is analyzed using the Random Forest models.
# MAGIC
# MAGIC Because Random Forest models capture non-linear relationships and feature interactions, they provide a reliable way to estimate the relative importance of each feature in predicting both:
# MAGIC - Excess readmission ratio (regression)
# MAGIC - High readmission risk (classification)
# MAGIC
# MAGIC This analysis helps identify the key drivers of hospital readmission outcomes.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8.1 Random Forest Regression Feature Importance
# MAGIC Feature importance from the Random Forest Regressor is used to identify which variables contribute most to predicting hospital excess readmission ratios.

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt

# Getting feature importances from the random forest regressormodel
rf_reg_importance = pd.DataFrame(
{
  "Feature": feature_cols,
  "Importance": rf_reg_model.feature_importances_
}
)

# Sorting the dataframe by importance
rf_reg_importance = rf_reg_importance.sort_values(
  by= "Importance", 
  ascending=False
  ).reset_index(drop=True)

# Displaying the dataframe
display(rf_reg_importance)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 8.1.1 Feature Importance Plot - Random Forest Regression 

# COMMAND ----------

# DBTITLE 1,Untitled
plt.figure(figsize=(10, 8))

plt.barh(
    rf_reg_importance["Feature"][:10][::-1],
    rf_reg_importance["Importance"][:10][::-1]
)

plt.title("Top 10 Feature Importance - Random Forest Regressor")
plt.xlabel("Importance Score")
plt.ylabel("Feature")

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8.2 Classification Feature Importance
# MAGIC
# MAGIC Feature importance from the Random Forest Classifier is used to identify which variables are most influential in classifying hospitals into high and low readmission groups.

# COMMAND ----------

# Classification feature importance
rf_clf_importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": rf_clf_model.feature_importances_
})

rf_clf_importance = rf_clf_importance.sort_values(
    by="importance",
    ascending=False
).reset_index(drop=True)

display(rf_clf_importance)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 8.2.1  Feature Importance Plot - Random Forest Classifier
# MAGIC
# MAGIC Feature importance from the Random Forest Classifier is used to identify which variables are most influential in classifying hospitals into high and low readmission groups.

# COMMAND ----------

plt.figure(figsize=(10, 6))

plt.barh(
    rf_clf_importance["feature"][:10][::-1],
    rf_clf_importance["importance"][:10][::-1],
    color = "orange"
)

plt.title("Top 10 Feature Importance - Random Forest Classifier")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8.3 Feature Importance Comparison Outputs:
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### 8.3.1 Feature Importance - Random Forest Regression Output Note:
# MAGIC The feature importance results from the Random Forest Regressor show that **readmission_gap is by far the most dominant predictor**, contributing over **83% of the model importance**.
# MAGIC
# MAGIC This indicates that the difference between predicted and expected readmission rates is the primary driver of the excess readmission ratio. This is expected, as the target variable itself is derived from CMS readmission calculations.
# MAGIC
# MAGIC Other important features include:
# MAGIC - expected_readmission_rate  
# MAGIC - predicted_readmission_rate  
# MAGIC
# MAGIC These variables also directly relate to how hospital performance is measured, reinforcing that CMS-derived metrics play a major role in predicting readmission outcomes.
# MAGIC
# MAGIC All remaining features such as patient volume, cost metrics, and hospital rating contribute very minimally to the model. This suggests that once the core readmission metrics are known, additional hospital characteristics provide limited incremental predictive value.
# MAGIC
# MAGIC Overall, the regression model is heavily driven by **core CMS readmission indicators**, with operational and geographic variables playing a secondary role.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 8.3.2 Feature Importance - Random Forest Classifier Output Note:
# MAGIC The feature importance results from the Random Forest Classifier show a similar pattern to the regression model, with **readmission_gap dominating the model**, contributing approximately **87% of total importance**.
# MAGIC
# MAGIC This confirms that the classification of hospitals into high and low readmission groups is primarily determined by how far a hospital deviates from expected readmission performance.
# MAGIC
# MAGIC In addition to readmission_gap, the following features contribute modestly:
# MAGIC - avg_unplanned_score  
# MAGIC - predicted_readmission_rate  
# MAGIC - expected_readmission_rate  
# MAGIC - unplanned_return_rate  
# MAGIC
# MAGIC These variables reflect hospital quality and patient outcome measures, indicating that both performance metrics and patient-level outcomes influence classification.
# MAGIC
# MAGIC Other features such as hospital rating, patient volume, and geographic indicators have relatively small contributions, suggesting they play a supporting role rather than being primary drivers.
# MAGIC
# MAGIC Overall, the classification model confirms that **readmission-related metrics are the strongest indicators of high readmission risk**, while operational and contextual features provide additional but limited signal.

# COMMAND ----------

# MAGIC %md
# MAGIC # 9 Key Findings & Insights
# MAGIC This analysis explored hospital readmission performance using CMS data, combining exploratory analysis, clustering, and machine learning models to better understand the drivers of hospital readmission outcomes.
# MAGIC
# MAGIC ### 1. Geographic Variation in Readmissions
# MAGIC
# MAGIC State-level analysis shows that hospital readmission ratios vary across the U.S., with certain states consistently exhibiting higher or lower readmission performance.
# MAGIC
# MAGIC The choropleth and bubble maps indicate that states with larger hospital representation provide more stable averages, while smaller states may show higher variability due to fewer hospitals.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 2. Hospital Segmentation Reveals Distinct Performance Groups
# MAGIC
# MAGIC Clustering analysis identified clear hospital segments based on operational, cost, and quality characteristics:
# MAGIC
# MAGIC - Large metropolitan hospitals with high patient volume and moderate performance  
# MAGIC - Typical community hospitals with average metrics  
# MAGIC - Underperforming hospitals with high readmission and low ratings  
# MAGIC - High-cost hospitals with elevated spending (MSPB)  
# MAGIC - High-quality hospitals with strong ratings and low readmission rates  
# MAGIC
# MAGIC These segments highlight that hospital performance is not uniform and can be grouped into meaningful categories.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 3. Strong Predictability of Readmission Outcomes
# MAGIC
# MAGIC Both regression and classification models achieved very strong performance:
# MAGIC
# MAGIC - Random Forest Regressor achieved near-perfect fit (R² ≈ 0.995)  
# MAGIC - Classification models achieved near-perfect accuracy and ROC-AUC  
# MAGIC
# MAGIC This indicates that hospital readmission outcomes can be predicted with high confidence using the available features.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 4. Readmission Metrics Are the Primary Drivers
# MAGIC
# MAGIC Feature importance analysis shows that **readmission_gap** is the dominant predictor in both regression and classification models, followed by:
# MAGIC
# MAGIC - Expected readmission rate  
# MAGIC - Predicted readmission rate  
# MAGIC - Average unplanned score  
# MAGIC
# MAGIC This suggests that CMS-derived readmission metrics carry the strongest signal in determining hospital performance.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 5. Limited Impact of Operational and Geographic Features
# MAGIC
# MAGIC While variables such as hospital rating, patient volume, rural classification, and geographic indicators contribute to the models, their impact is relatively small compared to readmission-specific metrics.
# MAGIC
# MAGIC This indicates that overall hospital performance is driven more by **measured readmission behavior** than by general hospital characteristics.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 6. Consistency Across Modeling Approaches
# MAGIC
# MAGIC The agreement between:
# MAGIC
# MAGIC - Regression models  
# MAGIC - Classification models  
# MAGIC - Clustering results  
# MAGIC
# MAGIC suggests that the patterns observed in the data are stable and consistent across different analytical methods.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 7. Interpretation and Practical Implications
# MAGIC
# MAGIC The results suggest that:
# MAGIC
# MAGIC - Hospital readmission performance is largely captured by CMS-defined metrics  
# MAGIC - High-performing hospitals consistently show lower readmission gaps and better quality indicators  
# MAGIC - Underperforming hospitals can be identified through a combination of readmission metrics and operational indicators  
# MAGIC
# MAGIC From a healthcare analytics perspective, this analysis can support:
# MAGIC
# MAGIC - Hospital performance benchmarking  
# MAGIC - Identification of high-risk hospitals  
# MAGIC - Data-driven quality improvement initiatives  
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Final Note
# MAGIC
# MAGIC The strong predictive performance observed in the models is expected, as several input features are directly related to the calculation of readmission metrics. This highlights the importance of understanding feature relationships when interpreting model results.

# COMMAND ----------

# MAGIC %md
# MAGIC # 10 Conclusion
# MAGIC This project explored hospital readmission performance using CMS data through an end-to-end analytics pipeline built in Databricks.
# MAGIC
# MAGIC The analysis combined data engineering (Bronze → Silver → Gold), exploratory data analysis, clustering, dimensionality reduction (PCA), and predictive modeling.
# MAGIC
# MAGIC From the modeling results, Random Forest models performed better than linear models for both regression and classification tasks, achieving higher accuracy and lower error metrics.
# MAGIC
# MAGIC Feature importance analysis showed that readmission-related variables, especially readmission gap and expected readmission rate, were the strongest predictors of hospital performance, while geographic and operational features had relatively smaller impact.
# MAGIC
# MAGIC Overall, the results show that readmission-related metrics are the strongest predictors of hospital performance. However, since these variables are derived from CMS calculations and closely tied to the target variable, they likely introduce information overlap rather than representing fully independent drivers.
# MAGIC - Future work could involve excluding CMS-derived variables and focusing on operational, demographic, and geographic features to better understand independent drivers of readmission risk.
# MAGIC
# MAGIC This project demonstrates how a lakehouse architecture combined with machine learning can be used to generate meaningful insights and predictive signals in healthcare data.
