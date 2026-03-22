# Databricks notebook source
# MAGIC %md
# MAGIC Quick check if;
# MAGIC * metadata columns exist
# MAGIC * column names sanitized
# MAGIC * rows look good
# MAGIC

# COMMAND ----------

# DBTITLE 1,yu8--u
from pyspark.sql import functions as F

spark.read.format("delta").load("/Volumes/workspace/cms_hospital/bronze/hrrp").where(F.col("state")=="TX").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Validation
# MAGIC Implement a custom Data Quality framework and quarantine records that fail critical validation.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Setup Paths and Imports

# COMMAND ----------

# Standard Spark imports for transformations and data types
from pyspark.sql import functions as F
from pyspark.sql  import types as T

# Configure Medallion layer paths in Unity Catalog
BRONZE = "/Volumes/workspace/cms_hospital/bronze"
SILVER = "/Volumes/workspace/cms_hospital/silver"
QUARANTINE = "/Volumes/workspace/cms_hospital/quarantine"

# Setup functions for Delta I/O
def read_delta(path: str):
  return spark.read.format("delta").load(path)

# write data to delta with overwrite and schema enforcement
def write_delta_overwrite(df, path: str):
  (
    df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .save(path)
  )
  

# COMMAND ----------

# MAGIC %md
# MAGIC #### Data Quality utilities (checks + report + quarantine)
# MAGIC * This creates a tiny Data Quality framework.
# MAGIC   * First we normalize zipcodes to 5-digit strings to ensure consistent joins across CMS datasets and RUCA rural-urban classification tables.

# COMMAND ----------

# Facility_ID normalization
def normalize_facility_id(df, col_name="facility_id", width=6):
    return (
        df
        .withColumn(col_name, F.col(col_name).cast("string"))
        # remove trailing .0 if present
        .withColumn(col_name, F.regexp_replace(F.col(col_name), r"\.0$", ""))
        # remove anything non-numeric just in case
        .withColumn(col_name, F.regexp_replace(F.col(col_name), r"[^0-9]", ""))
        # convert empty strings to null
        .withColumn(col_name, F.when(F.length(F.col(col_name)) > 0, F.col(col_name)).otherwise(F.lit(None)))
        # left-pad to 6 digits
        .withColumn(col_name, F.when(F.col(col_name).isNotNull(), F.lpad(F.col(col_name), width, "0")).otherwise(F.lit(None)))
    )

# COMMAND ----------

# Zipcode Normalization
from pyspark.sql import functions as F

def normalize_zip(df, col_name="zip_code"):
    return (
        df
        .withColumn(col_name, F.col(col_name).cast("string"))

        # Remove none-digit characters
        .withColumn(col_name, F.regexp_replace(F.col(col_name), r"[^0-9]", ""))

        # Keep first 5 digits if zipcode has more than 5 digits
        .withColumn(col_name, F.substring(F.col(col_name), 1, 5))

        # Restore leading zeros - pad to 5
        .withColumn(col_name, F.lpad(F.col(col_name), 5, "0"))

        # If zipcode not ends up with 5 digits, set to null
        .withColumn(col_name, F.when(F.col(col_name).rlike(r"^\d{5}$"), F.col(col_name)).otherwise(F.lit(None)))
    )

# COMMAND ----------


def dq_null_count(df, colname: str) -> int:
    return df.filter(F.col(colname).isNull()).count()

def dq_blank_count(df, colname: str) -> int:
    return df.filter(F.trim(F.col(colname)) == "").count()

def dq_invalid_zip_count(df, zip_col: str = "zip_code") -> int:
    return df.filter(F.col(zip_col).isNull()).count()    

def append_dq_row(rows, table_name, check_name, failed_rows, notes=""):
    rows.append((table_name, check_name, int(failed_rows), notes))

def dq_report_df(rows):
    return spark.createDataFrame(rows, schema="table string, check string, failed_rows int, notes string")

  

# COMMAND ----------

# MAGIC %md
# MAGIC #### Column Type Normalization
# MAGIC
# MAGIC

# COMMAND ----------

# Cleans currency/percentage strings (removes commas) and casts to Double
def to_double(df, colname: str):
    if colname not in df.columns:
        return df
    
    cleaned = F.trim(F.regexp_replace(F.col(colname).cast("string"), ",", ""))
    
    return df.withColumn(
        colname,
        F.when(
            cleaned.rlike(r"^-?\d+(\.\d+)?$"),
            cleaned.cast("double")
        ).otherwise(F.lit(None).cast("double"))
    )

# Cleans numeric strings and casts to Integer
def to_int(df, colname: str):
    if colname not in df.columns:
        return df
    
    cleaned = F.trim(F.regexp_replace(F.col(colname).cast("string"), ",", ""))

    return df.withColumn(
        colname,
        F.when(
            cleaned.rlike(r"^-?\d+$"),
            cleaned.cast("int")
        ).otherwise(F.lit(None).cast("int"))
    )

# Converts string column to proper date type
def to_date(df, colname: str):
    if colname not in df.columns:
        return df
    
    cleaned = F.trim(F.col(colname).cast("string"))
    
    return df.withColumn(
        colname,
        F.when(
            cleaned.rlike(r"^\d{2}/\d{2}/\d{4}$"),
            F.to_date(cleaned, "MM/dd/yyyy")
        ).when(
            cleaned.rlike(r"^\d{4}-\d{2}-\d{2}$"),
            F.to_date(cleaned, "yyyy-MM-dd")
        ).otherwise(F.lit(None).cast("date"))
    )


# COMMAND ----------

# MAGIC %md
# MAGIC #### Quarantine failed rows and saves them to a separate location for debugging.

# COMMAND ----------

def quarantine_rows(df, bad_condition, path: str):
  bad_df = df.filter(bad_condition)
  if bad_df.count() > 0:
    (bad_df.write.format("delta")
     .mode("append")
     .option("mergeSchema", "true")
     .save(path))    
  return bad_df.count()


# COMMAND ----------

# MAGIC %md
# MAGIC #### Load bronze tables

# COMMAND ----------

# Reading data from bronze tables using defined function
hrrp_bronze = read_delta(f"{BRONZE}/hrrp")
hosp_bronze = read_delta(f"{BRONZE}/hospital_info")
mspb_bronze = read_delta(f"{BRONZE}/mspb")
unplanned_bronze = read_delta(f"{BRONZE}/unplanned_visits")
ruca_bronze = read_delta(f"{BRONZE}/ruca")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Silver transformations
# MAGIC Converts Bronze (raw) → Silver (cleaned, conformed, deduplicated):
# MAGIC   * Type casting (string → numeric/date)
# MAGIC   * Normalization (ZIP codes, state codes)
# MAGIC   * Deduplication (latest ingestion per business key)
# MAGIC   * Business logic (RUCA bucketing)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### HRRP (Hospital Readmissions Reduction Program)

# COMMAND ----------

hrrp_bronze.columns

# COMMAND ----------

# DBTITLE 1,HRRP Readmission
# Cast ID/names to consistent types
hrrp_silver = hrrp_bronze
hrrp_silver = normalize_facility_id(hrrp_silver, "facility_id")
hrrp_silver = hrrp_silver.withColumn("measure_name", F.trim(F.col("measure_name")))

# Convert to consistent types
hrrp_silver = to_double(hrrp_silver, "excess_readmission_ratio")
hrrp_silver = to_double(hrrp_silver, "predicted_readmission_rate")
hrrp_silver = to_double(hrrp_silver, "expected_readmission_rate")
hrrp_silver = to_int(hrrp_silver, "number_of_discharges")
hrrp_silver = to_int(hrrp_silver, "number_of_readmissions")
hrrp_silver = to_date(hrrp_silver, "start_date")
hrrp_silver = to_date(hrrp_silver, "end_date")

# COMMAND ----------

# import window function
from pyspark.sql.window import Window

# Using window function to remove duplicates
hrrp_dedupe_window = (
  F.row_number().over(
    Window.partitionBy(
      "facility_id", "measure_name", "start_date", "end_date"
    )
    .orderBy(F.col("ingestion_ts").desc_nulls_last())
  )
)

# Remove duplicates of HRRP data
hrrp_silver = hrrp_silver.withColumn("_row_number_temp", hrrp_dedupe_window).filter(F.col("_row_number_temp") == 1).drop("_row_number_temp")


# COMMAND ----------

# MAGIC %md
# MAGIC #### Hospital General Info

# COMMAND ----------

hosp_bronze.columns

# COMMAND ----------

# Standardize IDs and state codes
hosp_silver = (
    hosp_bronze
    .transform(lambda df: normalize_facility_id(df, "facility_id"))
    .withColumn("state", F.upper(F.trim(F.col("state"))))
)

# Normalize ZIP codes greated than 5 digits)
hosp_silver = normalize_zip(hosp_silver, col_name="zip_code")

# Cast ratings (CMS strings → int; null if invalid)
hosp_silver = to_int(hosp_silver, "hospital_overall_rating")

# Deduplicate by latest ingestion partition by facility
hosp_dedupe_window = F.row_number().over(
    Window.partitionBy("facility_id").orderBy(F.col("ingestion_ts").desc_nulls_last())
)

# Remove duplicates of hospital data
hosp_silver = hosp_silver.withColumn("_row_number_temp", hosp_dedupe_window).filter(F.col("_row_number_temp") == 1).drop("_row_number_temp")

# COMMAND ----------

# MAGIC %md
# MAGIC #### MSPB (Medicare Spending Per Beneficiary)

# COMMAND ----------

mspb_bronze.columns

# COMMAND ----------

# Standardize facility IDs
mspb_silver = normalize_facility_id(mspb_bronze, "facility_id")

# Normalize ZIP for consistency across joins
mspb_silver = normalize_zip(mspb_silver, "zip_code")

# Cast fields that actually exist
mspb_silver = to_double(mspb_silver, "score")
mspb_silver = to_date(mspb_silver, "start_date")
mspb_silver = to_date(mspb_silver, "end_date")

# Deduplicate by latest ingestion per facility
part_cols = ["facility_id"] + (["fiscal_year"] if "fiscal_year" in mspb_silver.columns else [])

mspb_dedupe_window = F.row_number().over(
    Window.partitionBy(*part_cols).orderBy(F.col("ingestion_ts").desc_nulls_last())
)

mspb_silver = (
    mspb_silver
    .withColumn("_row_number_temp", mspb_dedupe_window)
    .filter(F.col("_row_number_temp") == 1)
    .drop("_row_number_temp")
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Unplanned Hospital Visits

# COMMAND ----------

unplanned_bronze.columns

# COMMAND ----------

# Standardize facility IDs
unplanned_visits_silver = (
    unplanned_bronze
    .transform(lambda df: normalize_facility_id(df, "facility_id"))
    .withColumn("measure_id", F.trim(F.col("measure_id")))
)

# Normalize ZIP + cast numerics/dates
unplanned_visits_silver = normalize_zip(unplanned_visits_silver, col_name=("zip_code"))

for c in ["score", "denominator", "number_of_patients", "number_of_patients_returned"]:
  unplanned_visits_silver = to_double(unplanned_visits_silver, c) if c == "score" else to_int(unplanned_visits_silver, c)

unplanned_visits_silver = to_date(unplanned_visits_silver, "start_date")
unplanned_visits_silver = to_date(unplanned_visits_silver, "end_date")

# Deduplicate by latest ingestion partition by facility and measure and date
part_cols_u = ["facility_id"] + (["measure_id"] if "measure_id" in unplanned_visits_silver.columns else []) + ["start_date", "end_date"]

unplannedVisit_dedupe_window = F.row_number().over(Window.partitionBy(*part_cols_u).orderBy(F.col("ingestion_ts").desc_nulls_last()))

unplanned_visits_silver = unplanned_visits_silver.withColumn("_row_number_temp", unplannedVisit_dedupe_window).filter(F.col("_row_number_temp") == 1).drop("_row_number_temp")

# COMMAND ----------

# MAGIC %md
# MAGIC #### RUCA (Rural-Urban Commuting Area) ZIP Classification

# COMMAND ----------

ruca_bronze.columns


# COMMAND ----------

ruca_bronze.select("zipcode", "state", "zipcodetype").show(14)

# COMMAND ----------

# Standardize RUCA ZIP and code columns
ruca_silver = (
    ruca_bronze
    .withColumnRenamed("zipcode", "zip_code")
    .withColumn("zip_code", F.col("zip_code").cast("string"))
    .withColumnRenamed("primaryruca", "ruca_code")
    .withColumnRenamed("secondaryruca", "secondary_ruca_code")
)

ruca_silver = normalize_zip(ruca_silver, col_name="zip_code")
ruca_silver = to_double(ruca_silver, "ruca_code")
ruca_silver = to_double(ruca_silver, "secondary_ruca_code")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Data quality validation
# MAGIC * Generates DQ summary + quarantines bad rows

# COMMAND ----------

# Generate DQ report for ruca_silverilver and append to DQ report

dq_rows = []

# HRRP checks
append_dq_row(dq_rows, "hrrp", "null_facility_id", dq_null_count(hrrp_silver, "facility_id"))
append_dq_row(dq_rows, "hrrp", "null_measure_name", dq_null_count(hrrp_silver, "measure_name"))
append_dq_row(dq_rows, "hrrp", "null_excess_readmission_ratio", dq_null_count(hrrp_silver, "excess_readmission_ratio"))

# Hospital info checks
append_dq_row(dq_rows, "hospital_info", "null_facility_id", dq_null_count(hosp_silver, "facility_id"))
append_dq_row(dq_rows, "hospital_info", "blank_state", dq_blank_count(hosp_silver, "state"))
if "zip_code" in hosp_silver.columns:
    append_dq_row(dq_rows, "hospital_info", "invalid_zip_code", dq_invalid_zip_count(hosp_silver, "zip_code"))

# MSPB checks
append_dq_row(dq_rows, "mspb", "null_facility_id", dq_null_count(mspb_silver, "facility_id"))
append_dq_row(dq_rows, "mspb", "null_score", dq_null_count(mspb_silver, "score"))
if "zip_code" in mspb_silver.columns:
    append_dq_row(dq_rows, "mspb", "invalid_zip_code", dq_invalid_zip_count(mspb_silver, "zip_code"))

# Unplanned visits checks
append_dq_row(dq_rows, "unplanned_visits", "null_facility_id", dq_null_count(unplanned_visits_silver, "facility_id"))
if "zip_code" in unplanned_visits_silver.columns:
    append_dq_row(dq_rows, "unplanned_visits", "invalid_zip_code", dq_invalid_zip_count(unplanned_visits_silver, "zip_code"))

# RUCA checks
if "zip_code" in ruca_silver.columns:
    append_dq_row(dq_rows, "ruca", "invalid_zip_code", dq_invalid_zip_count(ruca_silver, "zip_code"))

dq_df = dq_report_df(dq_rows)
display(dq_df.orderBy("table", "check"))

# COMMAND ----------

# MAGIC %md
# MAGIC Quarantine null facility ID rows

# COMMAND ----------

bad_hrrp_rows = quarantine_rows(hrrp_silver, F.col("facility_id").isNull(), f"{QUARANTINE}/hrrp_null_facility")

bad_hosp_rows = quarantine_rows(hosp_silver, F.col("facility_id").isNull(), f"{QUARANTINE}/hospital_info_null_facility")

bad_mspb_rows = quarantine_rows(mspb_silver, F.col("facility_id").isNull(), f"{QUARANTINE}/mspb_null_facility")

bad_unpl_rows = quarantine_rows(unplanned_visits_silver, F.col("facility_id").isNull(), f"{QUARANTINE}/unplanned_null_facility")


# COMMAND ----------

# validate quarantine counts
print("Quarantined rows:",
      {"hrrp_null": bad_hrrp_rows,
       "hospital_info_null": bad_hosp_rows,
       "mspb_null": bad_mspb_rows,
       "unplanned_null": bad_unpl_rows})

# COMMAND ----------

# MAGIC %md
# MAGIC #### Write clean silver tables

# COMMAND ----------

# Write cleaned data to silver tables
write_delta_overwrite(hrrp_silver, f"{SILVER}/hrrp_clean")
write_delta_overwrite(hosp_silver, f"{SILVER}/hospital_info_clean")
write_delta_overwrite(mspb_silver, f"{SILVER}/mspb_clean")
write_delta_overwrite(unplanned_visits_silver, f"{SILVER}/unplanned_visits_clean")
write_delta_overwrite(ruca_silver, f"{SILVER}/ruca_clean")

# COMMAND ----------

# Validate silver table counts
print("Silver HRRP:", read_delta(f"{SILVER}/hrrp_clean").count())
print("Silver Hospital:", read_delta(f"{SILVER}/hospital_info_clean").count())
print("Silver MSPB:", read_delta(f"{SILVER}/mspb_clean").count())
print("Silver Unplanned:", read_delta(f"{SILVER}/unplanned_visits_clean").count())
print("Silver RUCA:", read_delta(f"{SILVER}/ruca_clean").count())

# COMMAND ----------

print("HRRP distinct facility_id:", hrrp_silver.select("facility_id").distinct().count())
print("Hospital distinct facility_id:", hosp_silver.select("facility_id").distinct().count())
print("MSPB distinct facility_id:", mspb_silver.select("facility_id").distinct().count())
print("Unplanned distinct facility_id:", unplanned_visits_silver.select("facility_id").distinct().count())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Quality Validation Tables
# MAGIC
# MAGIC Several intermediate overlap datasets were created during the Silver layer to validate joins between CMS datasets and detect duplicate or overlapping hospital records.
# MAGIC
# MAGIC Examples include:
# MAGIC - hrrp_unplanned_overlap
# MAGIC - hrrp_mspb_overlap
# MAGIC - hrrp_hosp_overlap
# MAGIC
# MAGIC These tables were used for diagnostic and validation purposes during pipeline development and are not part of the final feature pipeline.

# COMMAND ----------

display(hrrp_silver.select("facility_id").distinct().orderBy("facility_id").limit(10))
display(hosp_silver.select("facility_id").distinct().orderBy("facility_id").limit(10))
display(mspb_silver.select("facility_id").distinct().orderBy("facility_id").limit(10))
display(unplanned_visits_silver.select("facility_id").distinct().orderBy("facility_id").limit(10))

# COMMAND ----------

hrrp_hosp_overlap = (
    hrrp_silver.select("facility_id").distinct()
    .join(hosp_silver.select("facility_id").distinct(), on="facility_id", how="inner")
    .count()
)

print("HRRP ↔ Hospital overlap:", hrrp_hosp_overlap)

# COMMAND ----------

hrrp_mspb_overlap = (
    hrrp_silver.select("facility_id").distinct()
    .join(mspb_silver.select("facility_id").distinct(), on="facility_id", how="inner")
    .count()
)

print("HRRP ↔ MSPB overlap:", hrrp_mspb_overlap)

# COMMAND ----------

hrrp_unplanned_overlap = (
    hrrp_silver.select("facility_id").distinct()
    .join(unplanned_visits_silver.select("facility_id").distinct(), on="facility_id", how="inner")
    .count()
)

print("HRRP ↔ Unplanned overlap:", hrrp_unplanned_overlap)
