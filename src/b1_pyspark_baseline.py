import time
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession, functions as F

import os
import sys

# 1. Correctif de compatibilité pour PySpark 3.5 + Java 21 sous Windows
java_options = (
    "-XX:+IgnoreUnrecognizedVMOptions "
    "--add-opens=java.base/java.lang=ALL-UNNAMED "
    "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED "
    "--add-opens=java.base/java.reflect=ALL-UNNAMED "
    "--add-opens=java.base/java.io=ALL-UNNAMED "
    "--add-opens=java.base/java.net=ALL-UNNAMED "
    "--add-opens=java.base/java.nio=ALL-UNNAMED "
    "--add-opens=java.base/java.util=ALL-UNNAMED "
    "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED"
)

os.environ['JAVA_TOOL_OPTIONS'] = java_options
os.environ['PYSPARK_SUBMIT_ARGS'] = f'--driver-java-options "{java_options}" pyspark-shell'


def create_spark_session():
    """Initialise une session PySpark contrôlée (Benchmark B1)."""
    return (
        SparkSession.builder
        .appName("MetroPT3_Benchmark_B1")
        .master("local[*]")  # Utilise les cœurs locaux de votre machine
        .config("spark.dynamicAllocation.enabled", "false")  # Autoscaling désactivé
        .config("spark.sql.shuffle.partitions", "16")       # Nombre de partitions fixe
        .getOrCreate()
    )

def process_metropt_pyspark(spark, input_path, output_path):
    """Traitement PySpark par fenêtres temporelles de 5 minutes (Traitement A)."""
    t0 = time.perf_counter()

    # 1. Lecture des données préparées (Parquet ou CSV)
    if input_path.endswith(".parquet"):
        df = spark.read.parquet(input_path)
    else:
        df = spark.read.csv(input_path, header=True, inferSchema=True)

    # 2. Conversion et formatage du timestamp
    df = df.withColumn("timestamp", F.to_timestamp("timestamp"))

    # 3. Agrégations par fenêtre de 5 minutes (Effectif, Moyenne, Max, Ecart-type population)
    result = df.groupBy(
        F.window("timestamp", "5 minutes")
    ).agg(
        F.count("TP2").alias("n"),
        F.avg("TP2").alias("mean"),
        F.max("TP2").alias("maximum"),
        F.stddev_pop("TP2").alias("std_pop")
    ).orderBy("window")

    # 4. Écriture pour forcer l'évaluation paresseuse de Spark
    result.write.mode("overwrite").parquet(output_path)

    duration = time.perf_counter() - t0
    print(f"✅ Traitement PySpark B1 terminé en {duration:.2f} secondes.")
    return duration

if __name__ == "__main__":
    spark = create_spark_session()
    # Adaptez avec les chemins réels de votre projet
    process_metropt_pyspark(spark, "C:/Users/HP/Desktop/bigdata/projet/SparkCode/data/MetroPT3(AirCompressor).csv", "results/b1_output.parquet")
    spark.stop()