& "$PSScriptRoot\.venv\Scripts\Activate.ps1"

$env:JAVA_HOME = "C:\Program Files\Java\jdk-21"
$env:Path = "C:\Program Files\Java\jdk-21\bin;" + $env:Path
$env:PYSPARK_PYTHON = (Get-Command python).Source
$env:PYSPARK_DRIVER_PYTHON = (Get-Command python).Source

python src/02_benchmark_b0_pandas_chunked.py
python src/03_benchmark_b1_spark_baseline.py
python src/04_benchmark_b2_spark_aqe.py
python src/05_benchmark_b3_spark_resource_only.py
python src/06_benchmark_b4_spark_joint_proposal.py
python src/07_compare_benchmarks.py