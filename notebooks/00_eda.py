# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "2"
# ///
# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Exploratory Data Analysis: CMS Hospital Performance
# MAGIC
# MAGIC This notebook explores CMS hospital quality and utilization data prepared in the Gold layer of the lakehouse pipeline. The analysis investigates patterns in hospital performance, readmission behavior, patient volume, and spending metrics. The goal is to identify meaningful relationships in the data and prepare features for downstream machine learning models that predict hospital readmission risk.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Objective:
# MAGIC * Explore the Gold ML feature table and perform exploratory data analysis to identify patterns in hospital performance, readmission behavior, patient utilization, and cost metrics. The analysis generates project-ready insights and visualizations that support downstream machine learning modeling.

# COMMAND ----------

# MAGIC %md
# MAGIC # 1 Dataset Overview

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

# DBTITLE 1,Quick schema
# Schema view for gold_ml
gold_ml.printSchema()
display(gold_ml.limit(20))

# COMMAND ----------

# Sanity check on the number of unique facility ids
print("gold_ml count:", gold_ml.count())
print("gold_ml unique facility ids: ",gold_ml.select("facility_id").distinct().count())

# COMMAND ----------

# MAGIC %md
# MAGIC # 2 Missingness Analysis

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
# MAGIC ## 2.1 Transpose Missing Values Table
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
# MAGIC # 3 Summary Analysis
# MAGIC Summary statistics for key numeric features (Hospital info)

# COMMAND ----------

# 1. Identify numeric columns using list comprehension of hospital_features
numeric_cols = [(c, t) for c, t in gold_hosp.dtypes if t in ('int', 'double', 'bigint', 'float')]

# 2. Print each numeric column's name and type
for c, t in numeric_cols:
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

# Show the summary statistics for the selected columns
display(
    gold_ml.select("excess_readmission_ratio").describe()
)

# COMMAND ----------

# MAGIC %md
# MAGIC # 4 Hospital Readmission Analysis

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.1 State Level Readmission Analysis

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.1.1 State Readmission Summary Table

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
# MAGIC ### 4.1.2 Choropleth Map: Readmission Ratio by State
# MAGIC To better understand geographic variation in hospital performance, a choropleth map was created using the average excess readmission ratio by state. This visualization highlights regional differences in hospital readmission burden across the United States.

# COMMAND ----------

# view the columns for the gold_ml spark DataFrame
gold_ml.columns

# COMMAND ----------

gold_ml.describe

# COMMAND ----------

import pyspark.sql.functions as F
# Create a DataFrame with state and average excess readmission ratio
state_df = (
    gold_ml
    .groupBy("state")
    .agg(
        F.avg("excess_readmission_ratio").alias("avg_excess_readmission_ratio"),
        F.count("*").alias("hospital_count")
    )
)

display(state_df)

# COMMAND ----------

# Convert the state_df to a pandas DataFrame
state_pd = state_df.toPandas()

# Create Choropleth map using plotly express
import plotly.express as px

fig = px.choropleth(
    state_pd,
    locations="state",
    locationmode="USA-states",
    color="avg_excess_readmission_ratio",
    scope="usa",
    color_continuous_scale="Viridis",
    hover_name="state",
    hover_data=["avg_excess_readmission_ratio"],
    labels={"avg_excess_readmission_ratio":"Avg Excess Readmission Ratio"},
    title="Average Excess Readmission Ratio by State"
)

# Add state labels to the map using scattergeo
fig.add_scattergeo(
    locations=state_pd["state"],
    locationmode="USA-states",
    text=state_pd["state"],
    mode="text",
    hoverinfo='skip',
    showlegend=False
)

fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **4.1.2 Choropleth Map Output:** The state-level maps show that hospital readmission performance varies across the United States. Some states have average excess readmission ratios above 1.0, indicating higher readmission burden relative to expectations, while others fall below 1.0. 
# MAGIC
# MAGIC **Note:** The bubble map also shows that states differ substantially in hospital count, which is important when interpreting the stability of state averages.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.1.3 Bubble Map: Hospital Readmission Ratio with Hospital Count by State
# MAGIC A complementary bubble map was also created to add context to the state-level readmission analysis. In this view, color represents the average excess readmission ratio and bubble size represents the number of hospitals contributing to each state’s average. This helps distinguish between states with broader hospital representation and states with relatively fewer observations.

# COMMAND ----------

# Create bubble map using plotly express for hospital count and excess readmission ratio by state
import plotly.express as px

fig = px.scatter_geo(
    state_pd,
    locations="state",
    locationmode="USA-states",
    size="hospital_count",
    color="avg_excess_readmission_ratio",
    scope="usa",
    color_continuous_scale="Viridis",
    size_max=28,   # slightly larger bubbles
    hover_name="state",
    hover_data={
        "hospital_count": True,
        "avg_excess_readmission_ratio": ":.3f"
    },
    title="Hospital Readmission Ratio with Hospital Count by State"
)

# Map appearance improvements
fig.update_geos(
    showcountries=False,
    showcoastlines=False,
    showland=True,
    landcolor="rgb(235, 235, 235)",
    lakecolor="white"
)

# Layout improvements
fig.update_layout(
    title_x=0.5,  # center title
    margin={"r":0,"t":50,"l":0,"b":0}
)

fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **4.1.3 Bubble Map Output:** The bubble map adds additional context to the state-level readmission analysis by incorporating hospital count into the visualization. While color represents the **average excess readmission ratio**, the bubble size indicates **how many hospitals** contributed to the state-level average.
# MAGIC
# MAGIC States with larger bubbles represent regions with a higher number of hospitals in the dataset, meaning the calculated averages are based on a broader sample of facilities. Conversely, smaller bubbles indicate states with fewer hospitals, where the average readmission ratio may be influenced by a smaller number of observations.
# MAGIC
# MAGIC This visualization helps interpret the reliability of the state-level averages and highlights states with both high readmission ratios and large hospital representation.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.2 Hospital Type Readmission Analysis

# COMMAND ----------

# Hospital type excess readmission ratio and pct high readmission summary
display(
    gold_ml.groupBy("hospital_type")
    .agg(
        F.count("*").alias("count"),
        F.avg("excess_readmission_ratio").alias("avg_err"),
        F.avg("high_readmission_flag").alias("pct_high_readmission")
    )
    .orderBy(F.desc("count"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.3 Hospital Ownership Readmission Analysis

# COMMAND ----------

# hospital ownership excess readmission ratio and pct high readmission summary
display(
    gold_ml.groupBy("hospital_ownership")
    .agg(
        F.count("*").alias("count"),
        F.avg("excess_readmission_ratio").alias("avg_err"),
        F.avg("high_readmission_flag").alias("pct_high_readmission")
    )
    .orderBy(F.desc("count"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC # 5 Rural vs Metro Comparison

# COMMAND ----------

# rural summary for rural hospitals
rural_summary = (
    gold_ml.groupBy("ruca_bucket")
    .agg(
        F.count("*").alias("row_count"),
        F.avg("excess_readmission_ratio").alias("avg_excess_readmission_ratio"),
        F.avg("high_readmission_flag").alias("pct_high_readmission"),
        F.avg("mspb_score").alias("avg_mspb_score"),
        F.avg("total_unplanned_patients").alias("avg_total_unplanned_patients")
    )
    .orderBy(F.desc("avg_excess_readmission_ratio"))
)

rural_summary.display()

# COMMAND ----------

# MAGIC %md
# MAGIC # 6 Feature Distributions & Skewness

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.1 Univariate Analysis
# MAGIC * Histograms showing the distribution of key target/predictor variables;
# MAGIC   * Excess Readmission Ratio
# MAGIC   * MSPB Score Distribution
# MAGIC   * Total Unplanned Patients

# COMMAND ----------

# Histogram of following features
display(gold_ml.select("excess_readmission_ratio"))
display(gold_ml.select("mspb_score"))
display(gold_ml.select("total_unplanned_patients"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.2 Hospital Rating Performance by Rurality
# MAGIC * Average excess readmission ratio grouped by star rating and rural/urban classification. Reveals if low-rated rural hospitals underperform.

# COMMAND ----------

# 
rating_summary = (
    gold_ml.groupBy("hospital_overall_rating", "ruca_bucket")
    .agg(
        F.count("*").alias("rows"),
        F.avg("excess_readmission_ratio").alias("avg_excess_readmit_ratio")
    )
    .orderBy("hospital_overall_rating", "ruca_bucket")
)

display(rating_summary)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.3 Skewness Assessment

# COMMAND ----------

# Skewness for gold_ml numeric features
from pyspark.sql.functions import skewness

gold_ml_skewness = gold_ml.select(
    skewness("predicted_readmission_rate").alias("predicted_readmission_rate"),
    skewness("expected_readmission_rate").alias("expected_readmission_rate"),
    skewness("excess_readmission_ratio").alias("excess_readmission_ratio"),
    skewness("hospital_overall_rating").alias("hospital_overall_rating"),
    skewness("mspb_score").alias("mspb_score"),
    skewness("avg_unplanned_score").alias("avg_unplanned_score"),
    skewness("total_unplanned_patients").alias("total_unplanned_patients"),
    skewness("unplanned_return_rate").alias("unplanned_return_rate"),
    skewness("readmission_gap").alias("readmission_gap")
)

display(gold_ml_skewness)

# COMMAND ----------

# MAGIC %md
# MAGIC # 7 Categorical & Segment Analysis

# COMMAND ----------

gold_hosp.select(F.col("ruca_code")).distinct().display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7.1 Bivariate Visualization
# MAGIC * Comparative Analysis (Scatter Plot using Pandas & Matplotlib + Jitter)
# MAGIC     * Jitter helps to distinguish discrete values by adding a tiny random horizontal offset.

# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt

corr_plot_pd = gold_ml.select(F.col("excess_readmission_ratio"),
                              F.col("hospital_overall_rating")).toPandas()


plot_df = corr_plot_pd.copy()
plot_df["hospital_rating_jitter"] = plot_df["hospital_overall_rating"] + np.random.uniform(-0.08, 0.08, len(plot_df))

plt.figure(figsize=(10, 6))
plt.scatter(
    plot_df["hospital_rating_jitter"],
    plot_df["excess_readmission_ratio"],
    alpha=0.35
)

plt.xlabel("Hospital Rating")
plt.ylabel("Excess Readmission Ratio")
plt.title("Hospital Rating vs Excess Readmission Ratio")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **7.1 Output:** Relationship Between Hospital Rating and Readmission Burden
# MAGIC
# MAGIC Hospitals with higher CMS overall ratings tend to exhibit lower excess readmission ratios.
# MAGIC Average readmission ratios decline steadily from rating 1 hospitals to rating 5 hospitals, suggesting that hospitals with stronger overall performance also demonstrate better readmission outcomes.

# COMMAND ----------

# MAGIC %md
# MAGIC # 8 Correlation Analysis

# COMMAND ----------

# MAGIC %md
# MAGIC Note: Not every hospital report measures such as HF, PN, AMI, etc. That's why the outcome variables may be missing by design and this is not a data quality but domain missingniss. Correlation analysis therefore uses pairwise available observations
# MAGIC rather than dropping rows globally.
# MAGIC
# MAGIC We remove following features from correlation analysis;
# MAGIC * Observed_readmission_rate -> 56% missing.
# MAGIC * Ruca_bucket -> This is a geographic classification code, not a continuous measure.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8.1 Pearson Correlation Matrix
# MAGIC * We use a custom PySpark SQL implementation to bypass environment-specific security restrictions.

# COMMAND ----------

# Check for null values in the selected columns
corr_cols = [
    "number_of_discharges",
    "number_of_readmissions",
    "predicted_readmission_rate",
    "expected_readmission_rate",
    "excess_readmission_ratio",
    "hospital_overall_rating",
    "mspb_score",
    "avg_unplanned_score",
    "total_unplanned_patients",
    "unplanned_return_rate",
    "ruca_code",
    "readmission_gap",
    "observed_readmission_rate"
]

# Check null counts for each selected column
corr_df = gold_hosp.select([
    F.sum(F.col(c).isNull().cast("int")).alias(c) for c in corr_cols
])
display(corr_df)

# COMMAND ----------

# Correlation between excess readmission ratio and mspb score
gold_hosp.stat.corr("excess_readmission_ratio", "mspb_score")

# COMMAND ----------

# Select columns for correlation analysis
corr_cols = [
    "number_of_discharges",
    "number_of_readmissions",
    "predicted_readmission_rate",
    "expected_readmission_rate",
    "excess_readmission_ratio",
    "hospital_overall_rating",
    "mspb_score",
    "avg_unplanned_score",
    "total_unplanned_patients",
    "unplanned_return_rate",
    "readmission_gap"
]

target = "excess_readmission_ratio"

for c in corr_cols:
    if c != target:
        corr = gold_hosp.stat.corr(target, c)
        print(f"{target} vs {c}: {corr}")

# COMMAND ----------

gold_hosp.columns

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8.2 Correlation Heatmap
# MAGIC * Correlation analysis using pandas and seaborn - Heatmap 

# COMMAND ----------

# Select columns for correlation analysis
corr_cols = [
    "predicted_readmission_rate",
    "expected_readmission_rate",
    "excess_readmission_ratio",
    "hospital_overall_rating",
    "mspb_score",
    "avg_unplanned_score",
    "total_unplanned_patients",
    "unplanned_return_rate",
    "readmission_gap"
]

# Convert the Spark DataFrame to a Pandas DataFrame
corr_pd = gold_ml.select(corr_cols).toPandas()
corr_pd

# COMMAND ----------

# Calculate the correlation matrix
corr_matrix = corr_pd.corr()
corr_matrix

# COMMAND ----------

# Plot the correlation matrix as a heatmap
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))

sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt= ".2f")
plt.title("Correlation Matrix of Hospital Features")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # 9 Unsupervised Learning: Hospital Segmentaion

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9.1 Feature Selection
# MAGIC
# MAGIC Used operational hospital metrics that capture utilization, cost, and outcomes.
# MAGIC
# MAGIC **Note:** Rows with missing values in the clustering features were excluded before fitting the K-Means model. In this dataset, gaps in CMS readmission measures come from how the metrics are defined and reported, not from random data loss, so imputing them could create artificial patterns and distort the resulting clusters.

# COMMAND ----------

# Setup clustering features and drop null values
from pyspark.sql import functions as F

cluster_features = [
    "excess_readmission_ratio",
    "mspb_score",
    "total_unplanned_patients",
    "unplanned_return_rate",
    "hospital_overall_rating",
    "avg_unplanned_score"
]

cluster_df = (
  gold_ml
  .select(*cluster_features)
  .dropna()
)

print("Rows used for clustering:", {cluster_df.count()},"\n")
display(cluster_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9.2 Assemble Feature Vector
# MAGIC * Using array_to_vector instead of VectorAssembler function as work around due to the strict Security Whitelisting policies of Databricks Serverless compute.

# COMMAND ----------

# Create a vector column from the selected features
from pyspark.ml.functions import array_to_vector

cluster_df_vector = (
  cluster_df
  .withColumn(
    "features_vector",
    array_to_vector(F.array(*[F.col(c) for c in cluster_features]))
  )
)


display(cluster_df_vector.select("features_vector"). limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9.3 Feature Scaling
# MAGIC * Perform manual scaling instead of ml.feature's StandardScaler due to white listing policy restrictions.

# COMMAND ----------

# Create vector column from the selected features
from pyspark.ml.functions import array_to_vector

cluster_features = [
    "excess_readmission_ratio",
    "mspb_score",
    "total_unplanned_patients",
    "unplanned_return_rate",
    "hospital_overall_rating",
    "avg_unplanned_score"
]

cluster_df = gold_ml.select(*cluster_features).dropna()

stats = cluster_df.select([
  F.mean(c).alias(f"{c}_mean") for c in cluster_features
] + [
F.stddev(c).alias(f"{c}_std") for c in cluster_features
]
).collect()[0].asDict()

display(stats)


# COMMAND ----------

# Scale the features using the mean and standard deviation
scaled_df = cluster_df

for c in cluster_features:
  mean_val = stats[f"{c}_mean"]
  std_val = stats[f"{c}_std"]
  scaled_df = scaled_df.withColumn(f"{c}_scaled", 
                                   (F.col(c)-F.lit(mean_val))/F.lit(std_val))
  
  scaled_cols = [f"{c}_scaled" for c in cluster_features]

  df_scaled = (
    scaled_df.withColumn(
      "features_vector",
      array_to_vector(F.array(*[F.col(c) for c in scaled_cols])
    )
  ))

display(df_scaled.select("features_vector").limit(10))


# COMMAND ----------

# MAGIC %md
# MAGIC ##  9.4 Optimal Cluster Selection
# MAGIC * Prepare clustering data in Pandas due to white listing policy restrictions.

# COMMAND ----------

# Convert the Spark DataFrame to a Pandas DataFrame with selected features
cluster_features = [
    "excess_readmission_ratio",
    "mspb_score",
    "total_unplanned_patients",
    "unplanned_return_rate",
    "hospital_overall_rating",
    "avg_unplanned_score"
]

# Convert the Spark DataFrame to a Pandas DataFrame with selected features
# NAs are dropped
cluster_pd = gold_ml.select(*cluster_features).dropna().toPandas()

print("Rows used for clustering:", len(cluster_pd))
cluster_pd.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 9.4.1 Standardize Features with Sklearn

# COMMAND ----------

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(cluster_pd[cluster_features])


# COMMAND ----------

print(display(X_scaled.shape))
print(display(X_scaled[:5]))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 9.4.2 Elbow Method

# COMMAND ----------

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

k_values = list(range(2, 11))
inertia_values = []

for k in k_values:
    kmeans = KMeans(n_clusters= k, random_state= 42, n_init= 10)
    kmeans.fit(X_scaled)
    inertia_values.append(kmeans.inertia_)

plt.figure(figsize= (8,6))
plt.plot(k_values, inertia_values, marker= 'o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **9.4.2 Output:** The elbow method was used to determine the optimal number of clusters by examining the within-cluster sum of squares (Inertia) across different values of k. The curve shows a sharp decrease in inertia up to approximately k = 5, after which the rate of improvement diminishes significantly. Therefore, k = 5 was selected as the optimal number of clusters for hospital segmentation.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 9.4.3 Silhoutte Score

# COMMAND ----------

from sklearn.metrics import silhouette_score

silhouette_score_list = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_score_list.append(silhouette_score(X_scaled, kmeans.labels_))


plt.figure(figsize=(8, 6))
plt.plot(range(2, 11), silhouette_score_list, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs Number of Clusters')
plt.show()

print("Silhouette Score`", list(silhouette_score_list))

# COMMAND ----------

# MAGIC %md
# MAGIC **9.4.3 Output:** The optimal number of clusters was determined using both the elbow method and silhouette analysis. The elbow plot flattened out around 5–6 clusters, meaning that adding more clusters after that only slightly reduced the within‑cluster variation.
# MAGIC
# MAGIC The silhouette score peaked at 2 clusters, which is typical but would split hospitals into segments that are too broad to be useful. After 3 clusters, the silhouette scores were fairly similar, indicating that cohesion did not change much as we increased k.
# MAGIC
# MAGIC Balancing these metrics with how interpretable the hospital segments are, we chose 5 clusters as the final solution.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9.5 K-Means Model

# COMMAND ----------

cluster_pd.columns

# COMMAND ----------

best_k = 5

kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
cluster_pd["cluster"] = kmeans_final.fit_predict(X_scaled)

cluster_pd[["cluster"] + cluster_features].head()

# COMMAND ----------

# Reminder cluster_features checked
cluster_features

# COMMAND ----------

# MAGIC %md
# MAGIC ### 9.5.1 Cluster Counts and Cluster Distribution Plot

# COMMAND ----------

# Value counts of clusters

cluster_counts = (
  cluster_pd["cluster"]
  .value_counts()
  .sort_index()
  .reset_index()
)

cluster_counts.columns = ["cluster", "count"]

cluster_counts

# COMMAND ----------

plt.figure(figsize=(8, 6))
plt.bar(cluster_counts["cluster"], cluster_counts["count"])
plt.xlabel("Cluster")
plt.ylabel("Count")
plt.title("Hospital Cluster Distribution")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9.6 Cluster Profiles

# COMMAND ----------

# MAGIC %md
# MAGIC ### 9.6.1 Cluster Profiles Tables

# COMMAND ----------

cluster_profiles = (
    cluster_pd
    .groupby("cluster")[cluster_features]
    .mean()
    .reset_index()
    .sort_values("cluster")
)

cluster_profiles

# COMMAND ----------

# MAGIC %md
# MAGIC ### 9.6.2 Cluster Individual Profile Plots

# COMMAND ----------

# Plot cluster profiles

for feature in cluster_features:
    plt.figure(figsize=(8, 6))
    plt.bar(cluster_profiles["cluster"], cluster_profiles[feature])
    plt.xlabel("Cluster")
    plt.ylabel(feature)
    plt.title(f"Cluster Profile: {feature}")
    plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC **9.6.2 Output:** Cluster Interpretation
# MAGIC
# MAGIC K-Means clustering produced five hospital segments. Since clustering algorithms assign numeric cluster IDs without inherent meaning, descriptive labels were derived by examining the average feature values within each cluster, including patient volume, readmission metrics, spending indicators, and hospital quality ratings.
# MAGIC
# MAGIC Based on the cluster profile analysis, the following hospital segments were identified:
# MAGIC
# MAGIC **Cluster 0 — Large Metropolitan Hospitals**
# MAGIC * Very high patient volume
# MAGIC * Good hospital ratings (~3.6)
# MAGIC * Moderate readmission ratios
# MAGIC
# MAGIC These hospitals likely represent large urban facilities handling high patient throughput.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Cluster 1 — Typical Community Hospitals**
# MAGIC * Medium patient volume
# MAGIC * Moderate hospital ratings (~3.2)
# MAGIC * Slightly elevated unplanned return rates
# MAGIC
# MAGIC This segment represents average regional hospitals with typical operational performance.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Cluster 2 — Underperforming Hospitals**
# MAGIC * Lowest hospital ratings
# MAGIC * Highest readmission ratios
# MAGIC * Highest unplanned return rates
# MAGIC
# MAGIC Hospitals in this cluster show the weakest performance across key readmission indicators.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Cluster 3 — Higher-Cost Hospitals**
# MAGIC * Highest Medicare Spending per Beneficiary (MSPB)
# MAGIC * Moderate readmission ratios
# MAGIC * Lower hospital ratings
# MAGIC
# MAGIC These hospitals appear to have higher spending levels relative to outcomes.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Cluster 4 — High-Quality Hospitals**
# MAGIC * Highest hospital ratings
# MAGIC * Lowest readmission ratios
# MAGIC * Lowest unplanned return rates
# MAGIC
# MAGIC This cluster represents the strongest-performing hospitals across the analyzed quality metrics.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 9.6.2.1

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9.7 PCA Visualization
# MAGIC
# MAGIC Principal Component Analysis (PCA) was applied to reduce the dimensionality of the clustering feature space while preserving the majority of variance. Two principal components were selected for visualization purposes, allowing cluster separation to be visualized in two-dimensional space while retaining most of the underlying data variability.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 9.7.1 Finding Best PCA Components

# COMMAND ----------

from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

pca = PCA()
pca.fit(X_scaled)

explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

print("Explained variance per component:")
print(explained_variance)

print("\nCumulative variance:")
print(cumulative_variance)

# COMMAND ----------

# MAGIC %md
# MAGIC **9.7.1 Output:** Principal Component Analysis (PCA) was used to examine how much variance each component captures from the clustering feature set.
# MAGIC
# MAGIC The first two principal components explain approximately **54.5% of the total variance**, while the first four components explain approximately **82.9% of the variance**.
# MAGIC
# MAGIC Although additional components capture more information, **two components were selected for visualization purposes**, allowing the cluster structure to be visualized in two-dimensional space.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC #### 9.7.1.1 Scree Plot
# MAGIC This plot helps with ncomponent cutoff or highest variance.

# COMMAND ----------

plt.figure(figsize=(8,5))

plt.plot(range(1, len(explained_variance)+1),
         cumulative_variance,
         marker='o')

plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance")

plt.axhline(y=0.80, color='r', linestyle='--')
plt.axhline(y=0.90, color='g', linestyle='--')

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 9.7.1.2 PCA Component Analysis

# COMMAND ----------

# View the pca components of the features in a dataframe
import pandas as pd

loadings = pd.DataFrame(
    pca.components_,
    columns=cluster_features
)

loadings

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### 9.7.2 PCA Plot  

# COMMAND ----------

# Check the first 2 rows of the dataframe
cluster_pd.head(2)

# COMMAND ----------

from sklearn.decomposition import PCA
import seaborn as sns

pca = PCA(n_components=2, random_state= 42)
X_scaled_pca = pca.fit_transform(X_scaled)

cluster_pd["pc1"] = X_scaled_pca[:, 0]
cluster_pd["pc2"] = X_scaled[:,1]

plt.figure(figsize=(8,6))
sns.scatterplot(
    data=cluster_pd,
    x="pc1",
    y="pc2",
    hue="cluster",
    palette="tab10",
    alpha=0.7
)

plt.title("Hospital Clusters (PCA Projection)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC **9.7.2 Output:** PCA Visualization of Hospital Clusters
# MAGIC * To better visualize the clustering results, Principal Component Analysis (PCA) was applied to reduce the feature space into two dimensions. This allows the hospital clusters identified by the K-Means model to be plotted and visually inspected.
# MAGIC The first two principal components explain approximately **54.5% of the total variance** in the dataset. While this does not capture all of the variance, it is sufficient to provide a meaningful two-dimensional representation of the hospital feature space.
# MAGIC * From the PCA projection, the clusters show identifiable patterns in how hospitals group together based on operational characteristics such as patient volume, readmission behavior, spending metrics, and overall hospital ratings. Some overlap between clusters is expected given the complexity of healthcare systems and the similarity of certain hospital profiles.
# MAGIC
# MAGIC Overall, the visualization supports the earlier cluster profiling results and helps illustrate how different hospital segments relate to one another in the feature space.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # 10 Key Findings
# MAGIC
# MAGIC The exploratory analysis of CMS hospital data reveals several meaningful patterns related to hospital performance, readmission behavior, patient volume, and cost structures.
# MAGIC
# MAGIC ### 1. Hospital Ratings and Readmission Performance
# MAGIC Hospitals with higher overall ratings generally show lower excess readmission ratios. While the relationship is not perfectly linear, the distribution indicates that higher-rated hospitals tend to manage patient outcomes more effectively, resulting in fewer readmissions relative to expected levels.
# MAGIC
# MAGIC ### 2. Rural vs Metropolitan Differences
# MAGIC Hospitals located in rural areas tend to have lower patient volumes compared to metropolitan hospitals. Rural hospitals also show greater variability in readmission performance, which may reflect differences in available resources, patient demographics, and healthcare accessibility.
# MAGIC
# MAGIC ### 3. Patient Volume Patterns
# MAGIC Large metropolitan hospitals handle significantly higher patient volumes, particularly in unplanned patient visits. However, higher patient volume does not necessarily translate to better performance, as some high-volume hospitals still exhibit elevated readmission ratios.
# MAGIC
# MAGIC ### 4. Cost and Utilization
# MAGIC The Medicare Spending per Beneficiary (MSPB) metric varies across hospitals and clusters. Some hospitals show relatively high spending without corresponding improvements in readmission outcomes, suggesting possible inefficiencies in healthcare delivery.
# MAGIC
# MAGIC ### 5. Hospital Segmentation
# MAGIC Using K-Means clustering, hospitals can be grouped into several distinct operational profiles. These clusters reflect differences in patient volume, hospital quality ratings, spending patterns, and readmission behavior.
# MAGIC
# MAGIC The clustering analysis identified five general hospital segments:
# MAGIC
# MAGIC • **Large metropolitan hospitals** – high patient volume with moderate readmission levels  
# MAGIC • **Typical community hospitals** – medium patient volume with average performance  
# MAGIC • **Underperforming hospitals** – lower ratings with higher readmission ratios  
# MAGIC • **Higher-cost hospitals** – elevated spending levels with moderate outcomes  
# MAGIC • **High-quality hospitals** – strong ratings with relatively low readmission rates  
# MAGIC
# MAGIC ### 6. Cluster Visualization
# MAGIC Principal Component Analysis (PCA) was used to project the hospital feature space into two dimensions. The PCA visualization confirms that the clusters represent meaningful groupings of hospital characteristics, although some overlap exists due to similarities among certain hospital profiles.
# MAGIC
# MAGIC ### 7. Implications for Modeling
# MAGIC These exploratory findings highlight several variables that may be useful predictors in downstream modeling tasks, including:
# MAGIC
# MAGIC - hospital overall rating
# MAGIC - MSPB spending score
# MAGIC - patient volume indicators
# MAGIC - unplanned return rate
# MAGIC - rural vs metropolitan classification
# MAGIC
# MAGIC These features will be used in the next stage of the project to build predictive models for hospital readmission performance.
# MAGIC
