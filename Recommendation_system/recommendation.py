import sys
import re
from pyspark.ml.recommendation import ALS
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql import *
from pyspark.sql.functions import monotonically_increasing_id
import math

spark = SparkSession.builder.appName("programming_1").getOrCreate()

def fitting_model(x,ALS):
	als = ALS(rank=8,maxIter=4,regParam=0.08, userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="nan")
	model= als.fit(x)
	return model

def write_to_file(r):
	f=open("sub_ratings.txt","w")
	for i in r:
		f.write(str(i)+"\n")
	"""
	o = r.rdd.flatMap(list)
	o.coalesce(1).saveAsTextFile("final_submission")
	"""

def print_ratings(a):
	ratings = []
	for r in a:
    		if (math.isnan(r)==True):
        		ratings.append(r)
    		else:
        		ratings.append(round(r))
	write_to_file(ratings)
	
def post_processing(pred):
	sorted_ratings= pred.sort(pred.index)
	p= sorted_ratings.select(sorted_ratings.index,sorted_ratings.movieId,sorted_ratings.userId,sorted_ratings.prediction)
	l = p.select("prediction").rdd.flatMap(list).collect()
	print_ratings(l)


#pre-processing
a = spark.read.text("train.dat").rdd
b = a.map(lambda row: row.value.split("\t"))
c = b.map(lambda a: Row(userId=int(a[0]),movieId=int(a[1]), rating=float(a[2]), timestamp=int(a[3])))
train = spark.createDataFrame(c)


d = spark.read.text("test.dat").rdd
e = d.map(lambda row: row.value.split("\t"))
f = e.map(lambda a: Row(userId=int(a[0]),movieId=int(a[1])))
g = spark.createDataFrame(f)
test = g.withColumn("index", monotonically_increasing_id())

#building model
model =fitting_model(train,ALS)
predicted_rating = model.transform(test)

#post_processing
post_processing(predicted_rating)
