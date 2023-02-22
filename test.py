import json
import pyspark
from pyspark.sql import SparkSession
from pyspark import SparkFiles
from pyspark.ml import Pipeline
from pyspark import SparkContext
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.clustering import BisectingKMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.clustering import BisectingKMeans
import pandas as pd
from pyspark.ml.feature import StringIndexer
import matplotlib.pyplot as plt
from pyspark.ml.regression import LinearRegression
from pyspark.ml.classification import LinearSVC
import time
import multiprocessing as mp
import psutil
import numpy as np


def run_operator(file_name,operator):

    conf=pyspark.SparkConf()
    # conf.set( "spark.serializer", "org.apache.spark.serializer.KryoSerializer" )
    conf.setAll([('spark.executor.memory', '16g'),('spark.driver.memory','16g'),('spark.executor.cores','6'),('spark.sql.shuffle.partitions','4'),('spark.ui.showConsoleProgress', 'true')])

    spark = SparkSession. \
        builder. \
        config(conf=conf). \
        appName("MLlib Clustering"). \
        getOrCreate()


    stages = []


    # upload file to Spark
    spark.sparkContext.addFile(file_name)

    df = spark.read.csv("file:///"+SparkFiles.get(file_name), header=True, inferSchema= True)
    df.limit(5).toPandas().head()
    df.persist(pyspark.StorageLevel.MEMORY_AND_DISK)
    # Transform string type columns to string indexer 
    idx = 0
    for col in df.columns:
        if col.startswith("string"):
            df =df.drop(col)
            idx += 1                

    print("indexing done")   
    assemble=VectorAssembler(inputCols=df.columns,outputCol = 'vector_column')
    stages.append(assemble)

    assembled_data=assemble.transform(df)

    print("pipeline done")

    start = time.time()

    match operator:
        case "k_means":
            cpu,ram = monitor(k_means(stages,df,assembled_data,assemble))
        case "linear_regression":
            cpu,ram = monitor(linear_regression(stages,df,assembled_data,assemble))
        case "linear_support_vector":        
            cpu,ram = monitor(linear_support_vector(stages,df,assembled_data,assemble))

    pipeline  = Pipeline(stages = stages)

    end = time.time()
    total_time_elapsed = end - start

    metrics = {}
    metrics["dataset"] = file_name
    metrics["total elapsed time"] = str(total_time_elapsed)

    min_cpu = max_cpu = cpu[0]
    min_ram = max_ram = ram[0]
    avg_cpu = avg_ram = 0

    for i in cpu:
        if i < min_cpu:
            min_cpu = i
        if i > max_cpu:
            max_cpu = i
        avg_cpu += i  

    for i in ram:
        if i < min_ram:
            min_ram = i
        if i > max_ram:
            max_ram = i

        avg_ram += i        

    avg_cpu /= len(cpu)
    avg_ram /= len(ram)

    metrics["min cpu usage"] = str(min_cpu)
    metrics["max cpu usage"] = str(max_cpu)
    metrics["avg cpu usage"] = str(avg_cpu)
    metrics["min ram usage"] = str(min_ram)
    metrics["max ram usage"] = str(max_ram)
    metrics["avg ram usage"] = str(avg_ram)
    
    return metrics



# k means
def k_means(stages,df,assembled_data,assemble):
    print("kmeans starting")
    silhouette_scores=[]
    evaluator = ClusteringEvaluator(featuresCol='vector_column',metricName='silhouette')
    stages.append(evaluator)
    for K in range(2,11):
        BKMeans_=BisectingKMeans(featuresCol='vector_column', k=K, minDivisibleClusterSize =1)
        stages.append(BKMeans_)
        BKMeans_fit=BKMeans_.fit(assembled_data)
        BKMeans_transform=BKMeans_fit.transform(assembled_data) 
        evaluation_score=evaluator.evaluate(BKMeans_transform)
        silhouette_scores.append(evaluation_score)

    fig, ax = plt.subplots(1,1, figsize =(10,8))
    ax.plot(range(2,11),silhouette_scores)
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('Silhouette Score')


# linear regression

def linear_regression(stages,df,assembled_data,assemble):
  train, test = df.randomSplit([0.7, 0.3])
  train = assemble.transform(train)
  test = assemble.transform(test)

  lr = LinearRegression(featuresCol='vector_column',
                      labelCol='int1') 
  stages.append(lr)                   
  lr = lr.fit(train)
  pred_train_df = lr.transform(train).withColumnRenamed('prediction','predicted_value')
#   pred_train_df.show(5)
  pred_test_df = lr.transform(test).withColumnRenamed('prediction', 'predicted_value')


# Linear Support Vector 

def linear_support_vector(stages,df,assembled_data,assemble):
  train, test = df.randomSplit([0.7, 0.3])
  train = assemble.transform(train)
  test = assemble.transform(test)
  lsvc = LinearSVC(featuresCol='vector_column', labelCol="bool1")
  stages.append(lsvc)
  lsvc = lsvc.fit(train)

  pred = lsvc.transform(test)


def monitor(target):
    worker_process = mp.Process(target=target)
    worker_process.start()
    p = psutil.Process(worker_process.pid)

    # log cpu usage of `worker_process` every 10 ms
    cpu_percents = []
    ram_percents = []
    while worker_process.is_alive():
        cpu_percents.append(p.cpu_percent())
        ram_percents.append(p.memory_percent())
        time.sleep(0.01)

    worker_process.join()
    return cpu_percents, ram_percents

if __name__ ==  '__main__':
    
    filename= "smaller.csv"
    with open ("metrics.json",'w') as f:
        m = [] 
        m.append(run_operator(filename,"linear_regression"))
        json.dump(m,f)