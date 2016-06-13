# SparkMachineLearning

Download test data from GitHub: https://github.com/apache/spark/tree/master/data/mllib

## RegressionExample
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

## CollaborativeFilteringExample
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

## NaiveBayesExample
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

## KMeansExample
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

## DecisionTreesClassificationExample
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

## DecisionTreesRegressionExample
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
