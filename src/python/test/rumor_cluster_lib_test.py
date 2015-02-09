#! /usr/bin/python

import rumor_cluster_lib as test_lib
from nltk.stem import PorterStemmer

print "Begin unit testing!"

tweet1_0 = '550848845370757120\tKurisu210\t137690133\t@fcdotasia haha I think i got it! lets see if they turn up..! Thanks!\tFri Jan 02 02:59:33 +0000 2015\ten\tSydney, Australia'

tweet1_1 = '550848845370757120\tKurisu210\t137690133\t@fcdotasia haha I think i got it! lets see if they turn up..! Thanks! !\tFri Jan 02 02:59:35 +0000 2015\ten'

tweet2 = '550848849531928576\tAPkrawczynski\t38251431\tWhat a block by Carl Landry.\tFri Jan 02 02:59:34 +0000 2015\ten\tMinneapolis'

tweet3 = '550848849527701504\tzxnybnmyxngzel\t2931566907\tRT @djMemphis10: Happy new year everyone! I am happy!\tFri Jan 02 02:59:34 +0000 2015\ten'

tweet4 = '550848849527701505\tzxnybnmyxngzel\t2931566907\tRT @djMemphis10: Happy chinese new year everyone! I am happy!\tFri Jan 02 02:59:34 +0000 2015\ten'

tweet5 = '550848849527701506\tzxnybnmyxngzel\t2931566907\tRT @djMemphis10: Happy mars new year everyone! I am happy!\tFri Jan 02 02:59:34 +0000 2015\ten'

tweet6 = '550848849527701507\tzxnybnmyxngzel\t2931566907\tRT @djMemphis10: What??! Happy mars new year everyone! I am happy!\tFri Jan 02 02:59:34 +0000 2015\ten'

tweet7 = '550848849527701508\tzxnybnmyxngzel\t2931566907\tRT @djMemphis10: Really??!! Happy mars new year everyone! I am happy!\tFri Jan 02 02:59:34 +0000 2015\ten'

tweet8 = '550848849527701509\tzxnybnmyxngzel\t2931566907\tRT @djMemphis10: Unconfirmed. Happy mars new year everyone! I am happy!\tFri Jan 02 02:59:34 +0000 2015\ten'

# TODO(zhezhao) Unit test function: zhe_pipeline.

# TODO(zhezhao) Unit test function: zhe_preprocess.

# Unit test function: ngram
print "Test Function: rumor_cluster_lib.ngram."
input_text = "this is test"
output = test_lib.ngram(input_text, 2, 2)
print "Output is: " + output.__str__() + " for input: " + input_text + "."

# Unit test function: euclid_sim
input_text2 = "this is test2"
fea2 = test_lib.ngram(input_text2, 2, 2)

print "Output is: " + test_lib.euclid_sim(output, fea2).__str__() + ", given input text1: " + input_text + ". and input text2: " + input_text2 + "."

# TODO(zhezhao) Unit test function: in_match.

# TODO(zhezhao) Unit test function: twitter_date_to_sec.

# Unit test for tweet init

t1 = test_lib.tweet(tweet1_0)
t2 = test_lib.tweet(tweet1_1)
t3 = test_lib.tweet(tweet2)
t4 = test_lib.tweet(tweet3)
t5 = test_lib.tweet(tweet4)
t6 = test_lib.tweet(tweet5)
t7 = test_lib.tweet(tweet6)
t8 = test_lib.tweet(tweet7)
t9 = test_lib.tweet(tweet8)

t1.check_format()

t1.generate_ngrams();

# Unit test for tweet_cluster.

test_cluster = test_lib.tweet_cluster([t1, t2])
test_cluster.insert_tweet(t3)
test_cluster.insert_tweet(t4)

test_cluster_2 = test_lib.tweet_cluster([t2, t4, t4])
test_cluster.merge_cluster(test_cluster_2)

test_cluster.distance(t1)
test_cluster_2.distance(t4)

# Unit test for rumor_detection_pipeline.
test_rd = test_lib.rumor_detection()
test_rd.new_tweet(t1)
test_rd.new_tweet(t2)
test_rd.new_tweet(t3)
test_rd.new_tweet(t4)
test_rd.new_tweet(t5)
test_rd.new_tweet(t6)
test_rd.new_tweet(t7)
test_rd.new_tweet(t8)
test_rd.new_tweet(t9)




