# Databricks notebook source
# MAGIC %md
# MAGIC ### Objective:
# MAGIC * Explore the Gold ML feature table and generate project-ready insights/visuals.

# COMMAND ----------

from pyspark.sql import functions as F

GOLD = "/Volumes/workspace/cms_hospital/gold"

def read_delta(path):
    return spark.read.format("delta").load(path)

gold_ml = read_delta(f"{GOLD}/ml_features")
gold_hosp = read_delta(f"{GOLD}/hospital_features")

print("gold_ml rows:", gold_ml.count())
print("gold_hosp rows:", gold_hosp.count)

# COMMAND ----------

# MAGIC %md
# MAGIC Quick schema

# COMMAND ----------

# DBTITLE 1,Quick schema
gold_ml.printSchema()
display(gold_ml.limit(20))

# COMMAND ----------

# Sanity check on the number of unique facility ids
print("gold_ml count:", gold_ml.count())
print("gold_ml unique facility ids: ",gold_ml.select("facility_id").distinct().count())

# COMMAND ----------

# MAGIC %md
# MAGIC Missing value summary

# COMMAND ----------

# missing values for each column in gold_ml
missing_df = gold_ml.select(
  [
    F.sum(F.col(c).isNull().cast("int")).alias(c) for c in gold_ml.columns
  ]
)

display(missing_df)

# COMMAND ----------

# MAGIC %md
# MAGIC Transpose missing values table
# MAGIC * Column-wise summary of null counts and percentages

# COMMAND ----------

missing_long = []

row = missing_df.collect()[0].asDict()
total_rows = gold_ml.count()

for col_name, null_count in row.items():
  missing_long.append((col_name, null_count, round(null_count/ total_rows, 4)))


missing_long_df = spark.createDataFrame(
  missing_long,
  ["col_name", "null_count", "null_pct"]
).orderBy(F.desc("null_count"))

display(missing_long_df)
                                        

# COMMAND ----------

# MAGIC %md
# MAGIC Target variable distribution

# COMMAND ----------

# examine the distribution of the excess readmission ratio
display(gold_ml.select("excess_readmission_ratio"))

# examine the distribution of the predicted readmission rate
display(
    gold_ml.groupBy("high_readmission_flag")
    .agg(F.count("*").alias("count"))
    .orderBy("high_readmission_flag")
    )

# COMMAND ----------

# MAGIC %md
# MAGIC Summary statistics for key numeric features (Hospital info)

# COMMAND ----------

# DBTITLE 1,Identify numeric columns in gold_hosp
# 1. Identify numeric columns using list comprehension of hospital_features
numeric_cols = [(c, t) for c, t in gold_hosp.dtypes if t in ('int', 'double', 'bigint', 'float')]

# 2. Print each numeric column's name and type
for c in numeric_cols:
    print(f"Column: {c}, Type: {t}")

# COMMAND ----------

# Assign the list of numeric columns to a variable with continuous values of hospital_features
numeric_cols = [
    "number_of_discharges",
    "number_of_readmissions",
    "predicted_readmission_rate",
    "expected_readmission_rate",
    "excess_readmission_ratio",
    "mspb_score",
    "avg_unplanned_score",
    "total_unplanned_patients",
    "unplanned_return_rate",
    "readmission_gap",
    "observed_readmission_rate",
    "hospital_overall_rating",
    "ruca_code"
]

# Print the summary statistics of the selected numeric columns
display(gold_hosp.select(numeric_cols).summary())

# COMMAND ----------

# MAGIC %md
# MAGIC State-level readmission summary

# COMMAND ----------

# State level readmission summary
state_summary = (
    gold_ml.groupBy("state")
    .agg(
        F.count("*").alias("row_count"),
        F.avg("excess_readmission_ratio").alias("avg_excess_readmission_ratio"),
        F.avg("predicted_readmission_rate").alias("avg_predicted_readmission_rate"),
        F.avg("expected_readmission_rate").alias("avg_expected_readmission_rate"),
        F.avg("mspb_score").alias("avg_mspb_score"),
        F.avg("total_unplanned_patients").alias("avg_total_unplanned_patients"),
        F.avg("high_readmission_flag").alias("pct_high_readmission")
    )
    .orderBy(F.desc("avg_excess_readmission_ratio"))
)

display(state_summary)

# COMMAND ----------

#Top 10 states with highest avg excess readmission ratio
#This is the deviation from expected performance
display(
    gold_ml.groupBy("state")
    .agg(F.avg("excess_readmission_ratio").alias("avg_excess_ratio"))
    .orderBy(F.desc("avg_excess_ratio"))
    .limit(10)
)

# COMMAND ----------

# MAGIC %md
# MAGIC Rural vs metro comparison

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC
