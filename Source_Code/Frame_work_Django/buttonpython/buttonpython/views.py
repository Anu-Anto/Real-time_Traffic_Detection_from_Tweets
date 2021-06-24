from django.shortcuts import render
from django.http import HttpResponse
from django.shortcuts import HttpResponse
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk import word_tokenize
from pandas import DataFrame
from geopy.geocoders import Nominatim
from datetime import date
from datetime import datetime
from django.utils import timezone
from time import sleep
import datetime as dt

locator = Nominatim(user_agent="myGeocoder")

import requests
import tweepy
import pandas as pd
import numpy as np
import time
import csv
import os
import re
import string
import pytz
import json

reconstructed_model = tf.keras.models.load_model("model.h5")

def index(request):
	return render(request,'home.html')

def output(request):
	#Fetching_data_for_real_time_predictions
	tweets=[]

	#Privet Keys
	consumer_key = 'Ker2ucA1np4j1OcInL9dbhSFl'
	consumer_secret = '3wv7a11NbH9cgVUkape4NG1UT1dfmPdbkhlHfAi1UynbyfMEfv'
	access_token = '1308079438936981504-IWnqkEN7LhpmRhZKsujF466LEJwQbn'
	access_token_secret = 'oRdbbtkWX4xeJuXEmaC8P8JyULFCACPXYfCt6MuCZ5Pj2'

	#Authentication
	auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_token, access_token_secret)
	api = tweepy.API(auth)

	#Querying_List
	qua=["traffic", "construction", "congestion", "incident","traffic jam"]
	
	#Geo_search_api
	places = api.geo_search(query="New York", granularity="city")
	place_id = places[0].id

	#TimeZone_Detection
	tz_NY = pytz.timezone('GMT') 
	datetime_NY = datetime.now(tz_NY)
	tz_NY = pytz.timezone('America/New_York') 
	current_time_New_York = datetime.now(tz_NY).time()
	datetime_New_York = current_time_New_York.strftime("%H:%M:%S")

	#Tweet_Fetching
	for i in qua:
		tweets += tweepy.Cursor(api.search, q='{} place:{}'.format(i, place_id),include_rts=False,since=str(date.today()), result_type="recent").items(10)

	try:
		if os.path.exists('tweet_collected.csv'):		
			os.remove('tweet_collected.csv')
		csvFile = open('tweet_collected.csv', 'a')
		csvWriter = csv.writer(csvFile)
		location=''
		data=[]
		Sl_No=-1
		#Tweet_Filtering
		for tweet in tweets:
			minute_created=(60*tweet.created_at.hour)+tweet.created_at.minute
			minute_timezone=(60*datetime_NY.hour)+datetime_NY.minute
			
			if tweet.place and minute_created>=(minute_timezone-55):
				data.append(tweet.text)
				loc= str(tweet.place.name)+(", New York")
				l = locator.geocode(str(loc))
				Sl_No+=1
				format = "%Y-%m-%d %H:%M:%S %Z%z"
				delta = dt.timedelta(hours = -4)
				now=tweet.created_at
				t = now.time()
				converted_time=(dt.datetime.combine(dt.date(1,1,1),t) + delta).time()

				delta = dt.timedelta(hours = 5, minutes=30)
				now_in=tweet.created_at
				t_in = now_in.time()
				converted_time_in=(dt.datetime.combine(dt.date(1,1,1),t_in) + delta).time()
				csvWriter.writerow([Sl_No,str(now.date()),str(datetime_New_York),str(converted_time),str(converted_time_in),tweet.id,l,tweet.text])
			else:
				continue

		csvFile.close()

	except BaseException as e:
		time.sleep(3)

	#Preprocessing
	num_words = 20000
	re.compile('<title>(.*)</title>')

	def remove_emoji(text):
	    emoji_pattern = re.compile("["u"\U0001F600-\U0001F64F"u"\U0001F300-\U0001F5FF"u"\U0001F680-\U0001F6FF"u"\U0001F1E0-\U0001F1FF"u"\U00002702-\U000027B0"u"\U000024C2-\U0001F251""]+", flags=re.UNICODE)
	    return emoji_pattern.sub(r'', str(text))

	def remove_punct(text):
	    text_nopunct = ''
	    text_nopunct = re.sub('['+string.punctuation+']', '', str(text))
	    return text_nopunct

	def remove_url(text):
	    url_pattern  = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
	    return url_pattern.sub(r'', str(text))

	def lower_token(tokens):
	    return [w.lower() for w in tokens]
	stoplist = stopwords.words('english')

	def removeStopWords(tokens):
    		return [word for word in tokens if word not in stoplist and word.isalpha()]


	tokenizer = Tokenizer(num_words=num_words,oov_token="unk")
	pre =pd.read_csv("tweet_collected.csv")
	pre_df =pd.DataFrame(pre)
	pre.columns = ['Sl_No','Date','Time_Ny_Now','Tweet_created_at_time_NY','Tweet_created_at_time_IN','ID','Location','Tweet']
	pre_df = pre_df.drop_duplicates()

	pre['Text'] = pre['Tweet'].apply(remove_emoji)
	pre['Text'] = pre['Tweet'].apply(remove_url)
	pre['Text'] = pre['Tweet'].apply(remove_punct)
	
	tokens_pre = [word_tokenize(sen) for sen in pre.Text]
	lower_tokens_pre = [lower_token(str(token)) for token in tokens_pre]
	filtered_words_pre = [removeStopWords(sen) for sen in lower_tokens_pre]
	pre['Text'] = [' '.join(sen) for sen in filtered_words_pre]
	

	tokenizer.fit_on_texts(pre['Text'].tolist())
	x_test  = np.array( tokenizer.texts_to_sequences(pre['Text'].tolist()) )
	x_test = pad_sequences(x_test, padding='post', maxlen=30)

	#Prediction
	y_pred = reconstructed_model.predict_proba(x_test)
	ans=y_pred.argmax(axis=1)
	pre['Prediction']= ans
	if os.path.exists('tweet_final.csv'):
		os.remove("tweet_final.csv")

	prediction = pd.DataFrame(pre, columns=['Sl_No','Date','Time_Ny_Now','Tweet_created_at_time_NY','Tweet_created_at_time_IN','ID','Location','Prediction','Tweet']).to_csv("tweet_final.csv",index=True)

	data = pd.read_csv("tweet_final.csv")

	json_records = data.reset_index().to_json(orient ='records')
	data = []
	data = json.loads(json_records)
	context = {'d': data}
	return render(request, 'table.html', context)

#Code For Future
#sorting
'''
data=datanotsort.sort_values(by=["Sl_No","ID","Location","Prediction","Tweet"], ascending=False)
'''
'''
and minute_created>=(minute_timezone-55)

New York
United States

celebrities
politicians
good
qua=["traffic"," blocked", "lane", "construction", "crash", "congestion", "delays", "vehicle", "incident", "ramp","street"]	location = New York'''
#sort assending 
'''data_nsort = pd.read_csv("tweet_collected.csv")
response = HttpResponse(content_type='text/csv')
data = data_nsort.sort_values(by=["prediction"], ascending=False)'''
#prediction with labels
'''
pre['prediction'] = np.where((pre.prediction == 0),'Non Traffic',pre.prediction)
pre['prediction'] = np.where((pre.prediction == '1'),'Traffic related',pre.prediction)
pre['prediction'] = np.where((pre.prediction == '2'),'Traffic Cleared',pre.prediction)
'''
#prediction anu
'''y_pred_keras = reconstructed_model.predict_proba(x_test).ravel()
	y_pred_keras = np.where(y_pred_keras > 0.9 ,1 ,0)
	y_pred_keras=pd.Series(y_pred_keras)
	pre['Prediction_anu']= y_pred_keras'''
#high threshold = high number of 0
#14 hr 30 min india a head
