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
import pyspark
# session
conf=pyspark.SparkConf()
# conf.set( "spark.serializer", "org.apache.spark.serializer.KryoSerializer" )
conf.setAll([('spark.executor.memory', '16g'),('spark.driver.memory','16g'),('spark.executor.cores','6'),('spark.sql.shuffle.partitions','4')])

spark = SparkSession. \
     builder. \
     config(conf=conf). \
     appName("MLlib Clustering"). \
     getOrCreate()

stages = []


# upload file to Spark
spark.sparkContext.addFile("Big_5_re.csv")

df = spark.read.csv("file:///"+SparkFiles.get("Big_5_re.csv"), header=True, inferSchema= True)
df.limit(5).toPandas().head()
df.persist(pyspark.StorageLevel.MEMORY_AND_DISK)
# Transform string type columns to string indexer 
idx = 0
for col in df.columns:
    if col.startswith("string"):
        # indexer = StringIndexer(inputCol=col, outputCol="string_col_"+str(idx))
        # stages.append(indexer)
        # df = indexer.fit(df).transform(df)
        df =df.drop(col)
        idx += 1                

print("indexing done")   
assemble=VectorAssembler(inputCols=df.columns,outputCol = 'vector_column')
stages.append(assemble)

assembled_data=assemble.transform(df)
# scaler?
# pipeline  = Pipeline(stages = stages)

print("pipeline done")

# k means
def k_means():
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
    plt.show()


k_means()


# linear regression

def linear_regression(df):
  train, test = df.randomSplit([0.7, 0.3])
  train = assemble.transform(train)
  test = assemble.transform(test)

  lr = LinearRegression(featuresCol='vector_column',
                      labelCol='int1') 
  stages.append(lr)                   
  lr = lr.fit(train)
  pred_train_df = lr.transform(train).withColumnRenamed('prediction','predicted_value')
  pred_train_df.show(5)
  pred_test_df = lr.transform(test).withColumnRenamed('prediction', 'predicted_value')
  pred_test_df.show(5)

# slinear_regression(df)  

# Linear Support Vector 

def linear_support_vector(df):
  
  train, test = df.randomSplit([0.7, 0.3])

  train = assemble.transform(train)
  test = assemble.transform(test)


  lsvc = LinearSVC(featuresCol='vector_column', labelCol="bool1")
  stages.append(lsvc)
  lsvc = lsvc.fit(train)

  pred = lsvc.transform(test)
  pred.show(3)

# linear_support_vector(df) 
pipeline  = Pipeline(stages = stages)