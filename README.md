# Big Data Benchmarking Pipeline (`SparkCode`)

This project covers two datasets processed in two different execution environments:

- **Small dataset (local execution):** Processed locally on your machine using the 5-minute Parquet pipeline and the benchmark suite (`B0`–`B4`).
- **Second dataset (notebook execution):** Processed in a notebook environment using Google Colab and Google Drive.

---

## 1. Small Dataset — Local Execution

The small dataset is processed locally using a Python virtual environment (`.venv`) and **JDK 21**.

Before running the project, ensure that the following environment variables are correctly configured:

- `JAVA_HOME` and `Path` point to JDK 21.
- `PYSPARK_PYTHON` points to the Python executable inside the virtual environment.
- `PYSPARK_DRIVER_PYTHON` points to the Python executable inside the virtual environment.

### Step 1: Prepare the Parquet Dataset

Before running the benchmarks, generate the prepared 5-minute Parquet dataset.

Run the following command from the project root directory:

```powershell
python src\01_prepare_metropt_parquet.py
```

Wait for the preparation process to finish before proceeding to the next step.

### Step 2: Run the Benchmarks

Once the Parquet dataset has been successfully generated, execute the benchmark suite (`B0`–`B4`) using the PowerShell script:

```powershell
.\run_spark.ps1
```

The script runs the Spark benchmarks on the prepared dataset.

---

## 2. Second Dataset — Notebook Execution

The second dataset is processed using a notebook environment.

Before running the notebook, ensure that you have at least **15 GB of available storage space in Google Drive** to accommodate the dataset and the associated processing files.

The notebook can then be executed in the configured Google Colab environment.

