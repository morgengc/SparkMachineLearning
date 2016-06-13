# SparkMachineLearning

Download external lib from: http://archive.apache.org/dist/spark/spark-1.6.0/spark-1.6.0-bin-hadoop2.6.tgz, in which `lib/spark-assembly-1.6.0-hadoop2.6.0.jar` is needed.

Download test data from GitHub: https://github.com/apache/spark/tree/master/data/mllib.

Compile this project in EclipseScala, generate `spark-machine-learning.jar`, and put this jar into cluster to run.

## 1. Classification & Regression
### NaiveBayesExample
```
spark-submit \
--class NaiveBayesExample \
--master spark://BigData1637:7077 \
--num-executors 6 \
--driver-memory 8g \
--executor-memory 2g \
--executor-cores 2 \
spark-machine-learning.jar > output
```

### DecisionTreesClassificationExample
```
spark-submit \
--class DecisionTreesClassificationExample \
--master spark://BigData1637:7077 \
--num-executors 6 \
--driver-memory 8g \
--executor-memory 2g \
--executor-cores 2 \
spark-machine-learning.jar > output
```

### RegressionExample
```
spark-submit \
--class RegressionExample \
--master spark://BigData1637:7077 \
--num-executors 6 \
--driver-memory 8g \
--executor-memory 2g \
--executor-cores 2 \
spark-machine-learning.jar > output
```

### DecisionTreesRegressionExample
```
spark-submit \
--class DecisionTreesRegressionExample \
--master spark://BigData1637:7077 \
--num-executors 6 \
--driver-memory 8g \
--executor-memory 2g \
--executor-cores 2 \
spark-machine-learning.jar > output
```

## 2. Clustering
### KMeansExample
```
spark-submit \
--class KMeansExample \
--master spark://BigData1637:7077 \
--num-executors 6 \
--driver-memory 8g \
--executor-memory 2g \
--executor-cores 2 \
spark-machine-learning.jar > output
```

## 3. Collaborative filtering
### CollaborativeFilteringExample
```
spark-submit \
--class CollaborativeFilteringExample \
--master spark://BigData1637:7077 \
--num-executors 6 \
--driver-memory 8g \
--executor-memory 2g \
--executor-cores 2 \
spark-machine-learning.jar > output
```
