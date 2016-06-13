import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext

object NaiveBayesExample {
	def main(args: Array[String]) {
		val conf = new SparkConf().setAppName("Spark Machine Learning: NaiveBayesExample")
		val sc = new SparkContext(conf)

		// 格式: 1,0 4 0
		val data = sc.textFile("hdfs://BigData1637:9000/input/spark-machine-learning/sample_naive_bayes_data.txt")
		
		// 变成标准格式: (1.0,[0.0,4.0,0.0])
		val parsedData = data.map { line =>
			val parts = line.split(',')
			LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(' ').map(_.toDouble)))
		}

		// 分隔为两个部分，60%的数据用于训练，40%的用于测试
		val splits = parsedData.randomSplit(Array(0.6, 0.4), seed = 11L)
		val training = splits(0)
		val test = splits(1)

		// 训练模型， Additive smoothing的值为1.0
		// model.labels里面保存的就是分类结果(0.0, 1.0, 2.0)
		val model = NaiveBayes.train(training, lambda = 1.0, modelType = "multinomial")
		print("labels: "); model.labels.foreach(a => print(a+" ")); print("\n")
		print("pi: "); model.pi.foreach(a => print(a+" ")); print("\n")
		print("theta: "); model.theta.foreach(a => {a.foreach(b => print(b+" ")); println(",")}); print("\n")

		// 用测试数据来验证模型的精度
		val predictionAndLabel = test.map(p => (model.predict(p.features), p.label))
		val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / test.count()
		println("Accurary = " + accuracy*100 + "%")
		
		// 预测类别
		println("Prediction of (0.5, 3.0, 0.5): " + model.predict(Vectors.dense("0.5, 3.0, 0.5".split(',').map(_.toDouble))))
		println("Prediction of (1.5, 0.4, 0.6): " + model.predict(Vectors.dense("1.5, 0.4, 0.6".split(',').map(_.toDouble))))
		println("Prediction of (0.3, 0.4, 2.6): " + model.predict(Vectors.dense("0.3, 0.4, 2.6".split(',').map(_.toDouble))))
	}
}
