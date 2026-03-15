# Databricks notebook source
# MAGIC %md
# MAGIC # Hospital Readmission Prediction Models
# MAGIC
# MAGIC This notebook builds predictive models using the Gold ML feature table prepared in the lakehouse pipeline.
# MAGIC
# MAGIC Two modeling tasks are explored:
# MAGIC
# MAGIC 1. **Regression:** Predict hospital excess readmission ratio.  
# MAGIC 2. **Classification:** Predict whether a hospital has a high readmission burden.
# MAGIC
# MAGIC # 1 Objective
# MAGIC The goal is to evaluate how well operational metrics, hospital quality indicators, spending measures, and geographic context can predict hospital readmission performance.
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
# MAGIC ##### **4 Modeling dataset size output note:** 
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

# Check target column stats
print(ml_pd[target_reg].describe())

# COMMAND ----------

# MAGIC %md
# MAGIC ##### **4 Modeling dataset target regression output note:** 
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



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------


