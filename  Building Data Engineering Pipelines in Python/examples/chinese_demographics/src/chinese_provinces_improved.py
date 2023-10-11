from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, sum


# Improved aggregation function, grouped by country and province
def aggregate_inhabitants_by_province(frame):
    return (frame
            .groupBy("country","province")
            .agg(sum(col("inhabitants")).alias("inhabitants"))
            )

