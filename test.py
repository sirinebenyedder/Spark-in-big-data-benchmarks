import pytest
from pyspark.sql import SparkSession
def build_spark(app_name: str, cores: int = 4, memory: str = "4g") -> SparkSession:
    return (
        SparkSession.builder.appName(app_name)
        .master(f"local[{cores}]")
        .config("spark.driver.memory", memory)
        .config("spark.eventLog.enabled", "false")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )