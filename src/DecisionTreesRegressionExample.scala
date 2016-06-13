import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext

object DecisionTreesRegressionExample {
	def main(args: Array[String]) {
		val conf = new SparkConf().setAppName("Spark Machine Learning: DecisionTreesRegressionExample")
		val sc = new SparkContext(conf)

		// Load and parse the data
		val data = MLUtils.loadLibSVMFile(sc, "hdfs://BigData1637:9000/input/spark-machine-learning/sample_libsvm_data.txt")
		// Split the data into training and test sets (30% held out for testing)
		val splits = data.randomSplit(Array(0.7, 0.3))
		val (trainingData, testData) = (splits(0), splits(1))

		// Train a DecisionTree model.
		//  Empty categoricalFeaturesInfo indicates all features are continuous.
		val categoricalFeaturesInfo = Map[Int, Int]()
		val impurity = "variance"
		val maxDepth = 5
		val maxBins = 32

		val model = DecisionTree.trainRegressor(trainingData, categoricalFeaturesInfo, impurity, maxDepth, maxBins)

		// Evaluate model on test instances and compute test error
		val labelsAndPredictions = testData.map { point =>
			val prediction = model.predict(point.features)
			(point.label, prediction)
		}
		val testMSE = labelsAndPredictions.map{ case (v, p) => math.pow(v - p, 2) }.mean()
		println("Test Mean Squared Error = " + testMSE)
		println("Learned regression tree model:\n" + model.toDebugString)
	}
}
