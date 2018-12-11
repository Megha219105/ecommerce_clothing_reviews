'''
The module is used for NLP. It contains function to convert csv into list of data and a dict of columns.
Next it has three parts:
1) Tokenize the texts and filter the words and gives frquency distribution. For the given data all reviews, reviews with rating less than 4
and reviews more than four were used to tokenize and filter. This is done to know the words which are mostly
used in three.
2) Sentiment analysis. The function gives the sentiment, polarity and text. For this data, 
each review text's sentimets and polarity are used to know the behaviour of
reviewer. And it was compared with the rating also.
3) Prediction. The function tokenize and filter the texts which can be used for prediction.
 '''

import csv
from collections import Counter
import statistics as st
import re
import nltk
from nltk.corpus import stopwords
from nltk import sent_tokenize, word_tokenize
from stop_words import get_stop_words
from textblob import TextBlob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report


STOP_WORDS = list(get_stop_words('en'))         
NLTK_WORDS = list(stopwords.words('english'))   
NLTK_WORDS.extend(STOP_WORDS)
TOP_N = 100

csv_file = open('data.csv', encoding = "utf8", newline = '')
data_csv = csv.reader(csv_file, delimiter=',')
csv_file.close()

def convert_csv_into_list(data_csv):
	'''For data in csv file is converted into a list of all values. Also an access dictionary is 
	drawn to access all columns of the data. This returns'''
	line_count = 0
	data = []
	access = {}
	for row in data_csv:
		if line_count == 0:
			start = 0
			for keys in row:
				access.update({keys:start})
				start += 1
			col_list = row
			line_count += 1
		else:
			line_count += 1
			data.append(row)
	return (data, access, col_list)


def convert_string_to_integer(data, access):
	'''returns the columns Positive Feedback Count, Rating and Age are converted to integers. The first 
	column is unnamed and gives the serial number. This is also converted to integers.'''
	for i, row in enumerate(data):
		index = [0, 2, 5, 7]
		for j in index:
			data[i][j] = int(data[i][j])


#Tokenize the texts and filter words.'''

def feq_dist_words(texts, n):
	'''Removes punctuation, tokenize the text and filter the words of given length.'''
	word_review = re.sub('[^A-Za-z]+', ' ', texts)
	word_tokens = word_tokenize(word_review)
	filtered_word = [w for w in word_tokens if not w in NLTK_WORDS and len(w) > n]
	most_common = Counter(filtered_word).most_common(TOP_N)
	word_dist = nltk.FreqDist(filtered_word)
	return word_dist

(data, access, col_list) = convert_csv_into_list(data_csv)
convert_string_to_integer(data, access)


def main():
	'''Main function to convert text into df of all, good ratings and bad ratings words and 
	their frequency.'''
	texts_1 = [row[access['Review Text']].lower() for row in data]
	texts = ' '.join(texts_1)
	word_list = feq_dist_words(texts, 2)
	val1 = pd.DataFrame(word_list.most_common(TOP_N), columns=['Word', 'Frequency'])
	texts_2 = [row[access['Review Text']] for row in data if row[access['Rating']] >= 4]
	texts_good = ' '.join(texts_2)
	word_list_good = feq_dist_words(texts_good, 2)
	val2 = pd.DataFrame(word_list_good.most_common(TOP_N), columns=['Word', 'Frequency'])
	texts_3 = [row[access['Review Text']] for row in data if row[access['Rating']] < 4]
	texts_bad = ' '.join(texts_3)
	word_list_bad = feq_dist_words(texts_bad, 2)
	val3 = pd.DataFrame(word_list_bad.most_common(TOP_N), columns=['Word', 'Frequency'])
	return val1, val2, val3


all_df, good_df, bad_df = main()
print("The most frequent words in all review text are, {0}".format(all_df[:10]))
print("The most frequent words in good review texts are, {0}".format(good_df[:10]))
print("The most frequent words in bad review texts are, {0}".format(bad_df[:10]))


#Sentiment Analysis

def blob_list(text):
	'''Creates an list of text, polarity, and subjectivity.'''
	bloblist_desc = list()
	for row in text:
    	blob = TextBlob(row)
    	bloblist_desc.append([row, blob.sentiment.polarity, blob.sentiment.subjectivity])
    return bloblist_desc


def sentiment_type(polarity_desc):
	'''converts sentiment from a quantitative number to qualitative review.'''
    if polarity_desc['Sentiment'] > 0:
        val = "Positive Review"
    elif polarity_desc['Sentiment'] == 0:
        val = "Neutral Review"
    else:
        val = "Negative Review"
    return val

def polarity_type(polarity_desc):
	'''Converts the polarity probability to a qualitative response.'''
    if polarity_desc['Polarity'] > 0.5:
        val = "Positive Review"
    else:
        val = "Negative Review"
    return val

##Predict Recommendation based on the review text

def tokenize(text):
	'''Tokenize the text and removes the punctuation.'''
	remove_punct = re.sub('[^A-Za-z]+', ' ', text)
	tokenize_words = word_tokenize(remove_punct)
	filter_words = ''
	for words in tokenize_words:
		if not words in NLTK_WORDS and len(words) > 2:
			filter_words += words
	return filter_words


#Sentiment Analysis
review_text = [row[access['Review Text']] for row in data]
bloblist_desc = blob_list(review_text)
polarity_desc = pd.DataFrame(bloblist_desc, columns = ['Review','Sentiment','Polarity'])
polarity_desc['Rating'] = [row[access['Rating']] for row in data]

polarity_desc['Sentiment Type'] = polarity_desc.apply(sentiment_type, axis=1)
polarity_desc['Polarity Type'] = polarity_desc.apply(polarity_type, axis=1)



##Predict Recommendation based on the review text

X_text = [tokenize(row[access['Review Text']].lower()) for row in data]
Y_recommend = [row[access['Recommended IND']] for row in data]

bow_transformer=CountVectorizer().fit(X_text)
#The above code forms the 2D matrix of words and the sentences. Its act as a feature extraction

X_text = bow_transformer.transform(X_text)
X_train, X_test, y_train, y_test = train_test_split(X_text, Y_recommend, test_size=0.3, random_state=101)
naive_bayes = MultinomialNB()
naive_bayes.fit(X_train, y_train)
predict_recommendation = naive_bayes.predict(X_test)
print(confusion_matrix(y_test, predict_recommendation))
print('\n')
print(classification_report(y_test, predict_recommendation))


