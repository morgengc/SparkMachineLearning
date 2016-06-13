import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.regression.LinearRegressionModel
import org.apache.spark.mllib.regression.LinearRegressionWithSGD
import org.apache.spark.mllib.regression.RidgeRegressionModel
import org.apache.spark.mllib.regression.RidgeRegressionWithSGD
import org.apache.spark.mllib.regression.LassoModel
import org.apache.spark.mllib.regression.LassoWithSGD
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext

object RegressionExample {
	def main(args: Array[String]) {
		val conf = new SparkConf().setAppName("Spark Machine Learning: RegressionExample")
		val sc = new SparkContext(conf)

		// 格式: 1, 2 3 4 5 6 7 8 9
		// 逗号之前是Label，后面8个是Features
		val data = sc.textFile("hdfs://BigData1637:9000/input/spark-machine-learning/lpsa.txt")
		
		// 变成标准格式：(1, [2 3 4 5 6 7 8 9]). 其中1就是Label, [2 3 4 5 6 7 8 9]就是Features
		// cache()的作用是缓存计算结果，避免下次调用时重复执行中间计算过程
		val parsedData = data.map { line =>
			val parts = line.split(',')
			LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(' ').map(_.toDouble)))
		}.cache()

		// 将样本数据划分训练样本(占80%)与测试样本(占20%)
		val splits = parsedData.randomSplit(Array(0.8, 0.2))
		val training = splits(0).cache()
		val test = splits(1).cache()
		val numTraining = training.count()
		val numTest = test.count()
		println("Training: " + numTraining + ", test:" + numTest)

		// 线性回归训练样本 
		val numIterations = 100     // 迭代次数
		val stepSize = 1            // 步长
		val miniBatchFraction = 1.0 // 迭代因子
		val model = LinearRegressionWithSGD.train(training, numIterations, stepSize, miniBatchFraction)
		
		// TODO 模型评估怎么做
		
		// model是LinearRegressionModel类型，查手册得知该类型有weights和intercept两个变量
		println("LinearRegressionWithSGD model weights: " + model.weights)
		println("LinearRegressionWithSGD model intercept: " + model.intercept)

		// 对测试样本进行测试
		val prediction = model.predict(test.map(_.features))        // 使用模型，对测试样本的feature进行计算，得到预测值
		val predictionAndLabel = prediction.zip(test.map(_.label))  // 一一组合预测值和label
		println("compare prediction and label:")
		val tmp = predictionAndLabel.collect()
		tmp.foreach(println)

		// 计算测试误差. loss是误差的平方和，RMSE是MSE的平方根
		val loss = predictionAndLabel.map { case(v, p) => math.pow((v-p), 2) }.reduce(_ + _)
		val rmse = math.sqrt(loss / numTest)
		println("Test RMSE = " + rmse)

		// 给定一个新的feature，计算Label
		var newdata = "1.0, 1.0, 2.0, 1.0, 3.0, -1.0, 1.0, -2.0"
		println("LinearRegressionModel predict = " + model.predict(Vectors.dense(newdata.split(',').map(_.toDouble))))
		
		println("------")

		// RidgeRegressionWithSGD
		val model1 = RidgeRegressionWithSGD.train(training, numIterations)
		println("RidgeRegressionWithSGD model weights: " + model1.weights)
		println("RidgeRegressionWithSGD model intercept: " + model1.intercept)
		
		// 对测试样本进行测试
		val prediction1 = model1.predict(test.map(_.features))        // 使用模型，对测试样本的feature进行计算，得到预测值
		val predictionAndLabel1 = prediction1.zip(test.map(_.label))  // 一一组合预测值和label
		println("compare prediction and label:")
		val tmp1 = predictionAndLabel1.collect()
		tmp1.foreach(println)

		// 计算测试误差. loss是误差的平方和，RMSE是MSE的平方根
		val loss1 = predictionAndLabel1.map { case(v, p) => math.pow((v-p), 2) }.reduce(_ + _)
		val rmse1 = math.sqrt(loss1 / numTest)
		println("Test RMSE = " + rmse1)
		
		println("RidgeRegressionWithSGD predict = " + model1.predict(Vectors.dense(newdata.split(',').map(_.toDouble))))
		
		println("------")

		// LassoWithSGD
		val model2 = LassoWithSGD.train(training, numIterations)
		println("LassoWithSGD model weights: " + model2.weights)
		println("LassoWithSGD model intercept: " + model2.intercept)
		
		// 对测试样本进行测试
		val prediction2 = model2.predict(test.map(_.features))        // 使用模型，对测试样本的feature进行计算，得到预测值
		val predictionAndLabel2 = prediction2.zip(test.map(_.label))  // 一一组合预测值和label
		println("compare prediction and label:")
		val tmp2 = predictionAndLabel2.collect()
		tmp2.foreach(println)

		// 计算测试误差. loss是误差的平方和，RMSE是MSE的平方根
		val loss2 = predictionAndLabel2.map { case(v, p) => math.pow((v-p), 2) }.reduce(_ + _)
		val rmse2 = math.sqrt(loss2 / numTest)
		println("Test RMSE = " + rmse2)
		
		println("LassoWithSGD predict = " + model2.predict(Vectors.dense(newdata.split(',').map(_.toDouble))))
	}
}

