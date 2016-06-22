import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext

/*
 * 通过研判几个指标，预测公司是否破产
 * 数据集下载: https://archive.ics.uci.edu/ml/datasets/qualitative_bankruptcy
 * 数据集包含250个实例，其中143个实例为非破产，107个破产实例
 * 鉴于此数据集，我们必须训练一个模型，它可以用来分类新的数据实例，这是一个典型的分类问题
 */
object LogisticRegressionExample {

	// 将指标量化
	def getDoubleValue( input:String ) : Double = {
		var result:Double = 0.0
		if (input == "P")  result = 3.0 
		if (input == "A")  result = 2.0
		if (input == "N")  result = 1.0
		if (input == "NB") result = 1.0
		if (input == "B")  result = 0.0
		return result
	}

	def main(args: Array[String]) {
		val conf = new SparkConf().setAppName("Spark Machine Learning: KMeansExample")
		val sc = new SparkContext(conf)

		// 数据格式: P,N,N,N,A,A,B
		// 前面六列为因素，分别为工业风险、 管理风险、 财务灵活性、 信誉、 竞争力、 经营风险
		// 最后一列为该公司破产与否
		// 由于每个因素均为定性因素，不好量化，因此给他们设定一个取值范围：
		// P - positive; A - average; N - negative
		val data = sc.textFile("hdfs://BigData1637:9000/input/spark-machine-learning/Qualitative_Bankruptcy.data.txt")

		val parsedData = data.map { line =>
			val parts = line.split(",")
			LabeledPoint(getDoubleValue(parts(6)), Vectors.dense(parts.slice(0,6).map(x => getDoubleValue(x))))
		}

		// 原始数据的60%作为训练数据，40%作为测试数据
		val splits = parsedData.randomSplit(Array(0.6, 0.4), seed = 11L)
		val trainingData = splits(0)
		val testData = splits(1)

		// 设置分类个数为2个(破产/非破产)
		val model = new LogisticRegressionWithLBFGS().setNumClasses(2).run(trainingData)

		val labelAndPreds = testData.map { point =>
			val prediction = model.predict(point.features)
			(point.label, prediction)
		}
		println("compare label and prediction:")
		val tmp = labelAndPreds.collect()
		tmp.foreach(println)

		// 模型出错率
		val trainErr = labelAndPreds.filter(r =>  r._1 != r._2).count.toDouble / testData.count
		println("trainErr = " + trainErr)
	}
}
