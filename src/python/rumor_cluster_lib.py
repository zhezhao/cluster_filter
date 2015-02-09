#! /usr/bin/python

import fileinput
import re
import time
import math
from nltk.stem import PorterStemmer

# preprocesses text with stemming.
def zhe_pipeline(text, stemmer):
	pre_text = re.sub('[^A-Za-z0-9#\?\!\'\"\@\(\)\:\$\%]+',' ',text)
	pre_text = re.sub('\?',' ? ',pre_text)
	pre_text = re.sub('\@[A-Za-z0-9_]*',' @user ',pre_text)
	pre_text = re.sub('\!',' ! ',pre_text)
	pre_text = re.sub('\(',' ( ',pre_text)
	pre_text = re.sub('\)',' ) ',pre_text)
	pre_text = re.sub('\"',' \" ',pre_text)
	pre_text = re.sub('\:',' : ',pre_text)
	pre_text = re.sub('\$',' $ ',pre_text)
	pre_text = re.sub('\%',' % ',pre_text)
	pre_text = re.sub(' [ ]*', ' ', pre_text)
	pre_text = pre_text.lower()
	out_text = stemmer.stem(pre_text)
	return out_text

# preprocesses text without stemming.
def zhe_preprocess(text):
	pre_text = re.sub('[^A-Za-z0-9#\?\!\'\"\@\(\)\:\$\%]+',' ',text)
	pre_text = re.sub('\?',' ? ',pre_text)
	pre_text = re.sub('\@[A-Za-z0-9_]*',' @user ',pre_text)
	pre_text = re.sub('\!',' ! ',pre_text)
	pre_text = re.sub('\(',' ( ',pre_text)
	pre_text = re.sub('\)',' ) ',pre_text)
	pre_text = re.sub('\"',' \" ',pre_text)
	pre_text = re.sub('\:',' : ',pre_text)
	pre_text = re.sub('\$',' $ ',pre_text)
	pre_text = re.sub('\%',' % ',pre_text)
	pre_text = re.sub(' [ ]*', ' ', pre_text)
	pre_text = pre_text.lower()
	return pre_text

# extracts vector of ngrams from text.
def ngram(text, n = 4, minlen = 7):
	if text is None:
		return None
	pre_text = re.sub('[^A-Za-z0-9_#\@]+', ' ', text)
	pre_text = re.sub(' [ ]*', ' ', pre_text)
	tokens = pre_text.split(' ')
	while tokens.__contains__(''):
		tokens.remove('')
	if n < 1:
		n = 1
	if tokens.__len__() < n or tokens.__len__() < minlen:
		return None
	ngrams={}
	for i in range(0, tokens.__len__() - n + 1):
		ngram_index = '+'.join(tokens[i : i + n]);
		if ngram_index in ngrams:
			ngrams[ngram_index] = ngrams[ngram_index] + 1.0
		else:
			ngrams[ngram_index] = 1.0
	return ngrams

# calculates Euclidean distance between vectors.
def euclid_sim(fea1, fea2):
	tempfea = dict(fea2)
	dist = 0
	for ngram in fea1:
		if ngram in fea2:
			tempfea[ngram] = fea2[ngram] - fea1[ngram];
		else:
			tempfea[ngram] = fea1[ngram];
	for ngram in tempfea:
		dist = dist + tempfea[ngram]*tempfea[ngram]
	return math.sqrt(dist)

# return text if text contains signal patterns.
def in_match(text):
	text = text.lower()
	m1 = re.search('is (this|it|that) true', text)
	m2 = re.search('wha[a]*t[\?!][\?!]*', text)
	m3 = re.search('(rumor|debunk|unconfirmed)', text)
	m4 = re.search('(that|this|it) is not true', text)
	m5 = re.search('(real|really)\?', text)
	m6 = re.search('guess what', text)
	if m1 is None and m2 is None and m3 is None and m4 is None and m5 is None:
		return None
	if m6 is not None:
		return None
	text = re.sub('is (this|it|that) true', '9494632128',text)
	text = re.sub('wha[a]*t[\?!][\?!]*', '9494632128',text)
	text = re.sub('(rumor|debunk|unconfirmed)', '9494632128',text)
	text = re.sub('(that|this|it) is not true', '9494632128',text)
	text = re.sub('really\?[\?!]*', '9494632128', text)
	text = re.sub('real\?[\?!]*', '9494632128', text)
	text = re.sub('RT ','9494632128',text)
	text = re.sub('9494632128','',text)
	return text

# change twitter format date to seconds since epoch.
def twitter_date_to_sec(text):
	if text is None:
		return time.time()
	try:
		secs = time.mktime(time.strptime(text,'%a %b %d %H:%M:%S +0000 %Y'))
	except:
		secs = 0
	return secs

# Setting model parameters
class param:
	# the n in ngram.
	n = 4
	# min length of a tweet.
	minlen = 7
	# max distance between tweet and center of a cluster to match that tweet to that cluster.
	max_distance = 4.0
	# max distance between signal tweet to center of a cluster to include that signal tweet to that signal cluster.
	max_signal_distance = 4.0
	# minimum number of signal tweets to create new candidate rumor cluster.
	min_signal_num = 3
	# maximum number of non-signal tweets matching in retrieve back.
	max_nsignal_tweets = 0

class tweet:
	tid = ''
	uname = ''
	uid = ''
	text = ''
	date = ''
	lang = ''
	location = ''
	sec = 0
	ngrams = []
	# initializes tweet.
	def __init__(self, input_tweet = ''):
		fields = input_tweet.split('\t')
		if len(fields) >= 5:
			self.tid = fields[0]
			self.uname = fields[1]
			self.uid = fields[2]
			self.text = fields[3]
			# text_preserve will not be changed.
			self.text_preserve = fields[3]
			self.date = fields[4]
			self.sec = twitter_date_to_sec(self.date)
		if len(fields) > 5:
			self.lang = fields[5]
			if len(fields) > 6:
				self.location = fields[6]
	# checks the correctness of tweet format.
	def check_format(self):
		if self.tid == '':
			return False
		if self.text == '':
			return False
		try:
			secs = time.mktime(time.strptime(self.date, '%a %b %d %H:%M:%S +0000 %Y'))
		except:
			return False
		return True
	def generate_ngrams(self, params = param()):
		in_matched = in_match(self.text)
		if in_matched is None:
			ngrams = ngram(zhe_preprocess(self.text), params.n, params.minlen)
		else:
			ngrams = ngram(zhe_preprocess(in_matched), params.n, params.minlen)
		if ngrams is None:
			self.ngrams = []
		else:
			self.ngrams = ngrams

class tweet_cluster:
	tweets = []
	center = {}
	statement = ''
	first_sec = 0
	last_sec = 0
	config = param()
	# initializes tweet cluster.
	def __init__(self, tweets = [], params = param()):
		self.tweets = tweets
		self.config = params
		if tweets != []:
			self.calculate_center()
			self.update_time()
	# updates first and last sec of tweets.
	def update_time(self):
		first_sec = time.time()
		last_sec = 0
		for item in self.tweets:
			sec = twitter_date_to_sec(item.date)
			if sec > 0:
				if first_sec > sec:
					first_sec = sec
				if last_sec < sec:
					last_sec = sec
		if last_sec != 0:
			self.first_sec = first_sec
			self.last_sec = last_sec
	# calculates center for cluster.
	def calculate_center(self):
		self.center = {}
		num_tweets = len(self.tweets)
		for item in self.tweets:
			item.generate_ngrams(self.config)
			features = item.ngrams 
			for feature in features:
				if feature in self.center:
					self.center[feature] = self.center[feature] + features[feature] / num_tweets
				else:
					self.center[feature] = features[feature] / num_tweets
	# inserts new tweets and update center and last_sec.
	def insert_tweet(self, new_tweet):
		if new_tweet.check_format():
			if new_tweet.ngrams == []:
				new_tweet.generate_ngrams(self.config)
			features = new_tweet.ngrams
			if features == []:
				return False
			self.tweets.append(new_tweet)
			num_tweets = len(self.tweets)
			for feature in self.center:
				self.center[feature] = self.center[feature] * (num_tweets - 1) / num_tweets
			for feature in features:
				if feature in self.center:
					self.center[feature] = self.center[feature] + features[feature] / num_tweets
				else:
					self.center[feature] = features[feature] / num_tweets
			sec = twitter_date_to_sec(new_tweet.date)
			if self.first_sec > sec or self.first_sec == 0:
				self.first_sec = sec
			if self.last_sec < sec:
				self.last_sec = sec
			return True
		else:
			return False
	# merges an input cluster.
	def merge_cluster(self, new_cluster):
		num_tweets = len(self.tweets)
		new_num_tweets = len(new_cluster.tweets)
		# merge tweets.
		self.tweets.extend(new_cluster.tweets)
		# update first_sec and last_sec.
		if self.first_sec > new_cluster.first_sec:
			self.first_sec = new_cluster.first_sec
		if self.last_sec < new_cluster.last_sec:
			self.last_sec = new_cluster.last_sec
		# update center.
		for feature in self.center:
			self.center[feature] = self.center[feature] * num_tweets / (num_tweets + new_num_tweets)
		for feature in new_cluster.center:
			if feature in self.center:
				self.center[feature] = self.center[feature] + new_cluster.center[feature] * new_num_tweets / (num_tweets + new_num_tweets)
			else:
				self.center[feature] = new_cluster.center[feature] * new_num_tweets / (num_tweets + new_num_tweets)
	# calculates distance between center and input tweet.
	def distance(self, new_tweet):
		if new_tweet.ngrams == []:
			new_tweet.generate_ngrams(self.config)
		features = new_tweet.ngrams
		if features == []:
			return -1
		else:
			return euclid_sim(self.center, features)

class cluster_pool:
	config = param()
	# intializes rumor_cluster_pool
	def __init__(self, params = param()):
		self.config = params
		self.clusters = []
	# verifies whether a new tweet matched to existing rumor clusters.
	def match(self, new_tweet):
		if new_tweet.ngrams == []:
			new_tweet.generate_ngrams(self.config)
		features = new_tweet.ngrams
		matched = []
		if features != []:
			for i in range(0,len(self.clusters)):
				dis = self.clusters[i].distance(new_tweet)
				if dis >= 0 and dis < self.config.max_distance:
					matched.append(i)
		return matched
	# merges a set of clusters into one new cluster.
	def merge_clusters(self, inds):
		if len(inds) < 1:
			return -1
		if len(inds) == 1:
			return inds[0]
		new_cluster = tweet_cluster()
		new_cluster.config = self.config
		ind_red = 0
		for ind in inds:
			if ind < len(self.clusters):
				new_cluster.merge_cluster(self.clusters.pop(ind - ind_red))
				ind_red = ind_red + 1
		self.clusters.append(new_cluster)
		return len(self.clusters) - 1

class rumor_detection:
	# initializes rumor_detection pipeline.
	nsignal_tweets = []
	def __init__(self, params = param()):
		self.config = params
		self.rumor_clusters = cluster_pool(self.config)
		self.signal_clusters = cluster_pool(self.config)
	# inserts new tweet into pipeline.
	def new_tweet(self, new_tweet):
		inds = self.rumor_clusters.match(new_tweet)
		if inds != []:
			insert_ind = self.rumor_clusters.merge_clusters(inds)
			self.rumor_clusters.clusters[insert_ind].insert_tweet(new_tweet)
		else:
			in_matched = in_match(new_tweet.text)
			if in_matched is None:
				new_tweet.generate_ngrams(self.config)
				self.nsignal_tweets.append(new_tweet)
			else:
				new_tweet.text = in_matched
				inds = self.signal_clusters.match(new_tweet)
				if inds != []:
					insert_ind = self.signal_clusters.merge_clusters(inds)
					self.signal_clusters.clusters[insert_ind].insert_tweet(new_tweet)
					if len(self.signal_clusters.clusters[insert_ind].tweets) >= self.config.min_signal_num:
						new_rumor_cluster = self.retrieve_back(insert_ind)
						self.rumor_clusters.clusters.append(new_rumor_cluster)
				else:
					new_signal_cluster = tweet_cluster([], self.config)
					new_signal_cluster.insert_tweet(new_tweet)
					self.signal_clusters.clusters.append(new_signal_cluster)
	# pops signal cluster and retrieve back from non-signal tweet pool.
	def retrieve_back(self, ind):
		new_cluster = self.signal_clusters.clusters.pop(ind)
		for item in self.nsignal_tweets[-self.config.max_nsignal_tweets:]:
			dis = new_cluster.distance(item)
			if dis < self.config.max_distance and dis >= 0:
				new_cluster.insert_tweet(item)
		return new_cluster

