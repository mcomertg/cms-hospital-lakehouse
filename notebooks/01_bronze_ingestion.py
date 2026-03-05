# Databricks notebook source
# MAGIC %md
# MAGIC # CMS Hospital Lakehouse: Bronze Ingestion
# MAGIC **Goal:** Ingest raw CSV data from Unity Catalog Volumes into Delta Lake.
# MAGIC **Strategy:** We implement a sanitization function to ensure column names are compliant with Delta Lake's strict schema enforcement.

# COMMAND ----------

# MAGIC %md
# MAGIC Setup unity catalog path

# COMMAND ----------

# Bronze layer ingestion
raw_data_path = '/Volumes/workspace/cms_hospital/cms_raw'
bronze_base_path = '/Volumes/workspace/cms_hospital/bronze'

# COMMAND ----------

# MAGIC %md
# MAGIC Data Ingestions;
# MAGIC * HRRP (Hospital Readmissions Reduction Program)
# MAGIC * Hospital General Info
# MAGIC * MSPB (Medicare Spending Per Beneficiary)
# MAGIC * Unplanned Hospital Visits & RUCA (Rural-Urban Commuting Area)

# COMMAND ----------

# Import functions to read and write data
from pyspark.sql import functions as F
# Define csv reading functions 
def read_csv(path):
    return (
        spark
        .read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(path)
        .withColumn("ingestion_ts", F.current_timestamp())
        .withColumn("batch_id", F.date_format(F.current_timestamp(), "yyyyMMddHHmmss"))
        .withColumn("source_file", F.col("_metadata.file_path"))
    )

# COMMAND ----------

# Read raw CSV files
hrrp_df = read_csv(f'{raw_data_path}/FY_2026_Hospital_Readmissions_Reduction_Program_Hospital.csv')
hospital_info_df = read_csv(f"{raw_data_path}/Hospital_General_Information.csv")
mspb_df = read_csv(f"{raw_data_path}/Medicare_Hospital_Spending_Per_Patient-Hospital.csv")
unplanned_visits_df = read_csv(f"{raw_data_path}/Unplanned_Hospital_Visits-Hospital.csv")
ruca_df = read_csv(f"{raw_data_path}/RUCA-codes-2020-zipcode.csv")

# The Sanitization Function for column names
import re

def clean_colname(c: str) -> str:
  c = c.strip().lower()
  # replace invalid chars
  c = re.sub(r"[ ,;{}\(\)\n\t=]+", "_", c)
  # replace anything not alphanumeric or underscore
  c = re.sub(r"[^a-z0-9_]", "_", c)
  # collapse multiple underscores
  c = re.sub(r"_+", "_", c)
  # strip leading underscores
  c = c.strip("_")
  return c

# rename columns after sanitization
def sanitize_columns(df):
  new_cols = [clean_colname(c) for c in df.columns]
  return df.toDF(*new_cols)

# Sanitize each DataFrame before writing Delta
hrrp_df_s = sanitize_columns(hrrp_df)
hospital_info_df_s = sanitize_columns(hospital_info_df)
mspb_df_s = sanitize_columns(mspb_df)
unplanned_visits_df_s = sanitize_columns(unplanned_visits_df)
ruca_df_s = sanitize_columns(ruca_df)

# Write as "Delta" (Standardizing the data format to Delta for ACID compliance)
(hrrp_df_s.write
  .format("delta")
  .mode("overwrite")
  .option("overwriteSchema", "true")
  .save(f"{bronze_base_path}/hrrp"))
  
(hospital_info_df_s.write.format("delta").mode("overwrite").option("overwriteSchema","true").save(f"{bronze_base_path}/hospital_info"))
(mspb_df_s.write.format("delta").mode("overwrite").option("overwriteSchema","true").save(f"{bronze_base_path}/mspb"))
(unplanned_visits_df_s.write.format("delta").mode("overwrite").option("overwriteSchema","true").save(f"{bronze_base_path}/unplanned_visits"))
(ruca_df_s.write.format("delta").mode("overwrite").option("overwriteSchema","true").save(f"{bronze_base_path}/ruca"))

# COMMAND ----------

display(hrrp_df.select("_metadata.file_path").limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC Sanity check - Column verification

# COMMAND ----------

print(hrrp_df.columns)
print(hrrp_df_s.columns)

# COMMAND ----------

# MAGIC %md
# MAGIC MSPB Ingestion

# COMMAND ----------

# Count the number of records in each bronze table
print("HRRP:", spark.read.format("delta").load(f"{bronze_base_path}/hrrp").count())
print("Hospital Info:", spark.read.format("delta").load(f"{bronze_base_path}/hospital_info").count())
print("MSPB:", spark.read.format("delta").load(f"{bronze_base_path}/mspb").count())
print("Unplanned Visits:", spark.read.format("delta").load(f"{bronze_base_path}/unplanned_visits").count())
print("RUCA:", spark.read.format("delta").load(f"{bronze_base_path}/ruca").count())

# COMMAND ----------


# view the first 5 records in the HRRP table
spark.read.format("delta").load(f"{bronze_base_path}/hrrp").show(5)

# COMMAND ----------

# view the first 5 metadata records in the HRRP table
spark.read.format("delta").load(f"{bronze_base_path}/hrrp").select("ingestion_ts","batch_id","source_file").show(5, truncate=False)

# COMMAND ----------

hrrp_bronze = spark.read.format("delta").load(f"{bronze_base_path}/hrrp")
print(hrrp_bronze.columns)

# COMMAND ----------

spark.read.format("delta").load(f"{bronze_base_path}/unplanned_visits").select("ingestion_ts","batch_id","source_file").show(2, truncate=False)
