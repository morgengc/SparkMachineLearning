import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext

/*
协同过滤ALS算法推荐过程如下：

加载数据到 ratings RDD，每行记录包括：user, product, rate
从 ratings 得到用户商品的数据集：(user, product)
使用ALS对 ratings 进行训练
通过 model 对用户商品进行预测评分：((user, product), rate)
从 ratings 得到用户商品的实际评分：((user, product), rate)
合并预测评分和实际评分的两个数据集，并求均方差 
 */

object CollaborativeFilteringExample {
	def main(args: Array[String]) {
		val conf = new SparkConf().setAppName("Spark Machine Learning: CollaborativeFilteringExample")
		val sc = new SparkContext(conf)

		// Load and parse the data
		val data = sc.textFile("hdfs://BigData1637:9000/input/spark-machine-learning/u.data")
		val ratings = data.map(_.split(' ') match { case Array(user, item, rate) =>
			Rating(user.toInt, item.toInt, rate.toDouble)
		})

		// Build the recommendation model using ALS
		val rank = 10
		val numIterations = 10
		val model = ALS.train(ratings, rank, numIterations, 0.01)

		// 从ratings中获得只包含用户和商品的数据集 
		val usersProducts = ratings.map { case Rating(user, product, rate) =>
			(user, product)
		}

		// 使用推荐模型对用户商品进行预测评分，得到预测评分的数据集
		val predictions =
			model.predict(usersProducts).map { case Rating(user, product, rate) =>
				((user, product), rate)
			}

		// 将真实评分数据集与预测评分数据集进行合并
		val ratesAndPreds = ratings.map { case Rating(user, product, rate) =>
			((user, product), rate)
		}.join(predictions)

		// 然后计算均方差，注意这里没有调用 math.sqrt方法
		val MSE = ratesAndPreds.map { case ((user, product), (r1, r2)) =>
			val err = (r1 - r2)
			err * err
		}.mean()
		println("Mean Squared Error = " + MSE)

		// 为每个用户进行推荐，推荐的结果可以以用户id为key，结果为value存入redis或者hbase中
		/*
		val users=data.map(_.split(",") match {  
			case Array(user, product, rate) => (user)  
		}).distinct().collect()
		//users: Array[String] = Array(4, 2, 3, 1)
		 * 
		 */
		val users=data.map(_.split(" ")).map(u => u(1).toInt).distinct().collect()

		// 这一句还有逻辑错误，只能打印第一个用户
		users.foreach(user => 
		{  
			//依次为用户推荐商品   
			var rs = model.recommendProducts(user.toInt, numIterations)  
			var value = ""  
			var key = 0  

			//拼接推荐结果
			rs.foreach(r => {  
				key = r.user  
				value = value + r.product + ":" + r.rating + ","  
			})  
			println(key.toString+"   " + value)  
		})
	}
}


/*
import scala.collection.mutable

import org.apache.log4j.{Level, Logger}
import scopt.OptionParser

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.rdd.RDD

/**
 * An example app for ALS on MovieLens data (http://grouplens.org/datasets/movielens/).
 * Run with
 * 
 * bin/run-example org.apache.spark.examples.mllib.MovieLensALS
 * 
 * A synthetic dataset in MovieLens format can be found at `data/mllib/sample_movielens_data.txt`.
 * If you use it as a template to create your own app, please use `spark-submit` to submit your app.
 */
object MovieLensALS {

  case class Params(
      input: String = null,
      kryo: Boolean = false,
      numIterations: Int = 20,
      lambda: Double = 1.0,
      rank: Int = 10,
      numUserBlocks: Int = -1,
      numProductBlocks: Int = -1,
      implicitPrefs: Boolean = false)

  def main(args: Array[String]) {
    val defaultParams = Params()

    val parser = new OptionParser[Params]("MovieLensALS") {
      head("MovieLensALS: an example app for ALS on MovieLens data.")
      opt[Int]("rank")
        .text(s"rank, default: ${defaultParams.rank}}")
        .action((x, c) => c.copy(rank = x))
      opt[Int]("numIterations")
        .text(s"number of iterations, default: ${defaultParams.numIterations}")
        .action((x, c) => c.copy(numIterations = x))
      opt[Double]("lambda")
        .text(s"lambda (smoothing constant), default: ${defaultParams.lambda}")
        .action((x, c) => c.copy(lambda = x))
      opt[Unit]("kryo")
        .text("use Kryo serialization")
        .action((_, c) => c.copy(kryo = true))
      opt[Int]("numUserBlocks")
        .text(s"number of user blocks, default: ${defaultParams.numUserBlocks} (auto)")
        .action((x, c) => c.copy(numUserBlocks = x))
      opt[Int]("numProductBlocks")
        .text(s"number of product blocks, default: ${defaultParams.numProductBlocks} (auto)")
        .action((x, c) => c.copy(numProductBlocks = x))
      opt[Unit]("implicitPrefs")
        .text("use implicit preference")
        .action((_, c) => c.copy(implicitPrefs = true))
      arg[String]("<input>")
        .required()
        .text("input paths to a MovieLens dataset of ratings")
        .action((x, c) => c.copy(input = x))
    }

    parser.parse(args, defaultParams).map { params =>
      run(params)
    } getOrElse {
      System.exit(1)
    }
  }

  def run(params: Params) {
    val conf = new SparkConf().setAppName(s"MovieLensALS with $params")
    if (params.kryo) {
      conf.registerKryoClasses(Array(classOf[mutable.BitSet], classOf[Rating]))
        .set("spark.kryoserializer.buffer.mb", "8")
    }
    val sc = new SparkContext(conf)

    Logger.getRootLogger.setLevel(Level.WARN)

    //是否显性反馈
    val implicitPrefs = params.implicitPrefs

    //数据集
    val ratings = sc.textFile(params.input).map { line =>
      val fields = line.split("::")
      if (implicitPrefs) {
        /*
         * MovieLens ratings are on a scale of 1-5:
         * 5: Must see
         * 4: Will enjoy
         * 3: It's okay
         * 2: Fairly bad
         * 1: Awful
         * So we should not recommend a movie if the predicted rating is less than 3.
         * To map ratings to confidence scores, we use
         * 5 -> 2.5, 4 -> 1.5, 3 -> 0.5, 2 -> -0.5, 1 -> -1.5. This mappings means unobserved
         * entries are generally between It's okay and Fairly bad.
         * The semantics of 0 in this expanded world of non-positive weights
         * are "the same as never having interacted at all".
         */
        Rating(fields(0).toInt, fields(1).toInt, fields(2).toDouble - 2.5)
      } else {
        Rating(fields(0).toInt, fields(1).toInt, fields(2).toDouble)
      }
    }.cache()

    val numRatings = ratings.count()
    val numUsers = ratings.map(_.user).distinct().count()
    val numMovies = ratings.map(_.product).distinct().count()

    println(s"Got $numRatings ratings from $numUsers users on $numMovies movies.")

    //拆分数据，80%为训练集，20%为测试集
    val splits = ratings.randomSplit(Array(0.8, 0.2))
    val training = splits(0).cache()
    val test = if (params.implicitPrefs) {
      /*
       * 0 means "don't know" and positive values mean "confident that the prediction should be 1".
       * Negative values means "confident that the prediction should be 0".
       * We have in this case used some kind of weighted RMSE. The weight is the absolute value of
       * the confidence. The error is the difference between prediction and either 1 or 0,
       * depending on whether r is positive or negative.
       */
      splits(1).map(x => Rating(x.user, x.product, if (x.rating > 0) 1.0 else 0.0))
    } else {
      splits(1)
    }.cache()

    val numTraining = training.count()
    val numTest = test.count()
    println(s"Training: $numTraining, test: $numTest.")

    ratings.unpersist(blocking = false)

    val start = System.currentTimeMillis()
    val model = new ALS()
      .setRank(params.rank)
      .setIterations(params.numIterations)
      .setLambda(params.lambda)
      .setImplicitPrefs(params.implicitPrefs)
      .setUserBlocks(params.numUserBlocks)
      .setProductBlocks(params.numProductBlocks)
      .run(training)
    val end = System.currentTimeMillis()

    println("Train Time = " + (end-start)*1.0/1000)

    val rmse = computeRmse(model, test, params.implicitPrefs)

    println(s"Train RMSE = " + computeRmse(model, training,params.implicitPrefs))
    println(s"Test RMSE = $rmse.")

    sc.stop()
  }

  /** Compute RMSE (Root Mean Squared Error). */
  def computeRmse(model: MatrixFactorizationModel, data: RDD[Rating], implicitPrefs: Boolean) = {

    def mapPredictedRating(r: Double) = if (implicitPrefs) math.max(math.min(r, 1.0), 0.0) else r

    val predictions: RDD[Rating] = model.predict(data.map(x => (x.user, x.product)))
    val predictionsAndRatings = predictions.map{ x =>
      ((x.user, x.product), mapPredictedRating(x.rating))
    }.join(data.map(x => ((x.user, x.product), x.rating))).values
    math.sqrt(predictionsAndRatings.map(x => (x._1 - x._2) * (x._1 - x._2)).mean())
  }
}
*/