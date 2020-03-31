---
title: "PySpark in Practice"
date: 2020-03-15
tags: [data science, pyspark]
excerpt: Data Science, pyspark
---

```python
from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local").appName("Luke HW")\
    .config("spark.some.config.option", "some-value")\
    .getOrCreate()
```


```python
df = spark.read.option("header", "true").csv("baby-names.csv")
```


```python
df.columns
```




    ['state', 'sex', 'year', 'name', 'count']




```python
df.show(20)
```

    +-----+---+----+---------+-----+
    |state|sex|year|     name|count|
    +-----+---+----+---------+-----+
    |   AK|  F|1910|     Mary|   14|
    |   AK|  F|1910|    Annie|   12|
    |   AK|  F|1910|     Anna|   10|
    |   AK|  F|1910| Margaret|    8|
    |   AK|  F|1910|    Helen|    7|
    |   AK|  F|1910|    Elsie|    6|
    |   AK|  F|1910|     Lucy|    6|
    |   AK|  F|1910|  Dorothy|    5|
    |   AK|  F|1911|     Mary|   12|
    |   AK|  F|1911| Margaret|    7|
    |   AK|  F|1911|     Ruth|    7|
    |   AK|  F|1911|    Annie|    6|
    |   AK|  F|1911|Elizabeth|    6|
    |   AK|  F|1911|    Helen|    6|
    |   AK|  F|1912|     Mary|    9|
    |   AK|  F|1912|    Elsie|    8|
    |   AK|  F|1912|    Agnes|    7|
    |   AK|  F|1912|     Anna|    7|
    |   AK|  F|1912|    Helen|    7|
    |   AK|  F|1912|   Louise|    7|
    +-----+---+----+---------+-----+
    only showing top 20 rows




```python
df.count()
```




    5933561




```python
dfJohn = df.filter((df.name=="John"))
dfJohn.count()
```




    7018




```python
dfSex = df.filter((df.sex=="M"))
dfSexState = dfSex.filter((dfSex.state=="NE"))
dfSexStateYear = dfSexState.filter((dfSexState.year=="1980"))
dfSexStateYear.show(10)
```

    +-----+---+----+-----------+-----+
    |state|sex|year|       name|count|
    +-----+---+----+-----------+-----+
    |   NE|  M|1980|    Matthew|  434|
    |   NE|  M|1980|    Michael|  426|
    |   NE|  M|1980|      Jason|  409|
    |   NE|  M|1980|     Joshua|  366|
    |   NE|  M|1980|Christopher|  359|
    |   NE|  M|1980|     Justin|  337|
    |   NE|  M|1980|       Ryan|  320|
    |   NE|  M|1980|      David|  292|
    |   NE|  M|1980|     Andrew|  281|
    |   NE|  M|1980|      Brian|  278|
    +-----+---+----+-----------+-----+
    only showing top 10 rows




```python

```
