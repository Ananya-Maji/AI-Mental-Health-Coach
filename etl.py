from pyspark.sql import SparkSession
from pyspark.sql.functions import col, length, count

spark = SparkSession.builder \
    .appName("MentalHealthETL") \
    .getOrCreate()

# Read CSV data
df = spark.read.csv(
    "data/conversations.csv",
    header=True,
    inferSchema=True
)

print("===== ORIGINAL DATA =====")
df.show(truncate=False)

# Remove missing values
df = df.dropna(
    subset=["session_id", "user_input", "bot_response"]
)

# Calculate bot response length
df = df.withColumn(
    "length_of_response",
    length(col("bot_response"))
)

# Count number of turns in each session
turn_counts = df.groupBy("session_id").agg(
    count("turn_id").alias("number_of_turns_per_session")
)

# Join turn count with original data
df = df.join(
    turn_counts,
    on="session_id",
    how="left"
)

print("===== PROCESSED DATA =====")

df.select(
    "session_id",
    "turn_id",
    "user_input",
    "length_of_response",
    "number_of_turns_per_session"
).show(truncate=False)

# Save processed data
df.write.mode("overwrite").option(
    "header", "true"
).csv(
    "output/processed_conversations"
)
print("===== ETL COMPLETED SUCCESSFULLY =====")

spark.stop()