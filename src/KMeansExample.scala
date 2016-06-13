import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext

object KMeansExample {
	def main(args: Array[String]) {
		val conf = new SparkConf().setAppName("Spark Machine Learning: KMeansExample")
		val sc = new SparkContext(conf)

		// Load and parse the data
		val data = sc.textFile("hdfs://BigData1637:9000/input/spark-machine-learning/kmeans_data.txt")
		val parsedData = data.map(s => Vectors.dense(s.split(' ').map(_.toDouble))).cache()

		// Cluster the data into two classes using KMeans
		val numClusters = 2
		val numIterations = 20
		val clusters = KMeans.train(parsedData, numClusters, numIterations)

		/*
		// 计算测试数据分别属于那个簇类
		println(parsedData.map(v -> v.toString() + " belong to cluster :" + clusters.predict(v)));
		 */

		// Evaluate clustering by computing Within Set Sum of Squared Errors
		val WSSSE = clusters.computeCost(parsedData)
		println("Within Set Sum of Squared Errors = " + WSSSE)

		/*
		// 打印出中心点
		System.out.println("Cluster centers:");
		for (Vector center : clusters.clusterCenters()) {
			System.out.println(" " + center);
		}
		*/

		// 进行一些预测
		println("Prediction of (1.1, 2.1, 3.1): " + clusters.predict(Vectors.dense("1.1, 2.1, 3.1".split(',').map(_.toDouble))))
		println("Prediction of (10.1, 9.1, 11.1): " + clusters.predict(Vectors.dense("10.1, 9.1, 11.1".split(',').map(_.toDouble))))
		println("Prediction of (21.1, 17.1, 16.1): " + clusters.predict(Vectors.dense("21.1, 17.1, 16.1".split(',').map(_.toDouble))))
	}
}
