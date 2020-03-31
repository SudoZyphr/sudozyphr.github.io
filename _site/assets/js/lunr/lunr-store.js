var store = [{
        "title": "PySpark in Practice",
        "excerpt":"from pyspark.sql import SparkSession spark = SparkSession.builder.master(\"local\").appName(\"Luke HW\")\\ .config(\"spark.some.config.option\", \"some-value\")\\ .getOrCreate() df = spark.read.option(\"header\", \"true\").csv(\"baby-names.csv\") df.columns ['state', 'sex', 'year', 'name', 'count'] df.show(20) +-----+---+----+---------+-----+ |state|sex|year| name|count| +-----+---+----+---------+-----+ | AK| F|1910| Mary| 14| | AK| F|1910| Annie| 12| | AK| F|1910| Anna| 10| | AK| F|1910| Margaret| 8| | AK| F|1910| Helen|...","categories": [],
        "tags": ["data science","pyspark"],
        "url": "http://localhost:4000/PySparkExample/",
        "teaser": null
      },{
        "title": "Coronavirus in the United States",
        "excerpt":"Today we are faced with a virus that is rapidly spreading and affecting everyone across the world. While the world governments are taking action and implementing safety procedures; we still see that the virus is spreading. This draws the questions if these actions are preventing the spread or simply just...","categories": [],
        "tags": ["data science","random forest","prediction"],
        "url": "http://localhost:4000/covid19/",
        "teaser": null
      }]
