import org.apache.spark.mllib.fpm.FPGrowth
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext

object FPGrowthExample {

	def main(args: Array[String]) {
		val conf = new SparkConf().setAppName("Spark Machine Learning: FPGrowthExample")
		val sc = new SparkContext(conf)
		val baskets = sc.textFile("hdfs://BigData1637:9000/input/spark-machine-learning/sample_fpgrowth.txt").map { line =>
			line.split(" ")
		}

		// 设置支持度，目的是筛选掉一部分数据
		val fpGrowth = new FPGrowth().setMinSupport(0.5)

		val model = fpGrowth.run(baskets)
		
		// 打印频繁项集
		println(s"Number of frequent itemsets: ${model.freqItemsets.count()}")
		
		model.freqItemsets.collect().foreach { itemset =>
		  println(itemset.items.mkString("[", ",", "]") + ", " + itemset.freq)
		}

		// 给定一个置信度，生成关联关系
		val rules = model.generateAssociationRules(0.8)

		println("\nReglas obtenidas:")
		rules.foreach { rule =>
			println(rule.antecedent.toList + " -> " + rule.consequent.toList + ", confidence " + rule.confidence)
		}
	}

}
