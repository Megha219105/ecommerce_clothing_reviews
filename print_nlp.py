'''This contains all the print functions.'''

def print_summary(data, access, col_list):
	'''something to add....'''
	col_name = ', '.join(col_list) #row 23
	print("Column names are {}".format(col_name[1:]))
	print("Processed {} lines.".format(len(data)))
	print('The following are the columns with type:')
	for key in access:
		print('{0:35} type is : {1}'.format(key, type(data[0][access[key]])))
	


def print_after_changes(data, access, col_list):
	'''After changing the data columns into quantitative data.'''
	print('Now the type of quantative data has changed to:')
	for key in access:
		print('{0:35} type is : {1}'.format(key, type(data[0][access[key]])))

### Deparment which is reviewed the most

depart_reviewed = [row[access['Department Name']] for row in data if len(row[access['Review Text']]) > 0]		
print('The number of reviews in the data is,',len(depart_reviewed),'out of 23486')

mode_of_depart = st.mode(depart_reviewed)
print('The department which is reviewd the most is', mode_of_depart)
#Tops 

'''Counter({'Tops': 10048, 'Dresses': 6145, 'Bottoms': 3662, 'Intimate': 1653, 'Jackets': 1002, 
'Trend': 118, 'Others': 13})
'''

#Age groups that provide reviews
ages = [row[access['Age']] for row in data]
ages_str = []
#(min_age, max_age) = (min(ages), max(ages))
print('Minimum age is', min(ages),'and maximum age of the reviewer is', max(ages))
age_groups = ['Below 20', '20-30', '30-40', '40-50', '50-60', '60-70', 'Above 70']

for i, row in enumerate(data):
	if len(row[access['Review Text']]) > 0:
		if data[i][2] < 20:
			ages_str.append(age_groups[0])
		elif 20 <= data[i][2] < 30:
			ages_str.append(age_groups[1])
		elif 30 <= data[i][2] < 40:
			ages_str.append(age_groups[2])
		elif 40 <= data[i][2] < 50:
			ages_str.append(age_groups[3])
		elif 50 <= data[i][2] < 60:
			ages_str.append(age_groups[4])
		elif 60 <= data[i][2] < 70:
			ages_str.append(age_groups[5])
		else:
			ages_str.append(age_groups[6])
			

ages_cnt = Counter(ages_str)
print(ages_cnt)
print('The age group that reviewd the most is 30-40 followed by 40-50')
'''Counter({'30-40': 7346, '40-50': 5903, '50-60': 3834, '20-30': 2795, '60-70': 2256, 'Above 70': 463, 'Below 20': 44})'''

#Add extra column which review length
access['Review Length'] = 12
for i, row in enumerate(data):
	row.append(len(row['Reveiw Text']))
#Box plot of review length and rating

##Analyse all the reviews first:

#from nltk.tokenize import RegexpTokenizer

#from wordcloud import WordCloud, STOPWORDS






#Texts of good reviews, mean of ratings is 4.1, so we split the data into good reviews and bad reviews.
#Texts of good reviews >=4 and bad reviews < 4
texts_2 = [row[access['Review Text']] for row in data if row[access['Rating']] >= 4]
#18208 are good reviews
texts_good = ' '.join(texts_2)
#Remove Punctuations
word_review_good = re.sub('[^A-Za-z]+', ' ', texts_good)         

word_tokens_good = word_tokenize(word_review_good)
filtered_word_good = [w for w in word_tokens_good if not w in nltk_words and len(w) >2]
most_good = Counter(filtered_word_good).most_common(top_N)
print('Words which are often used in positive reviews are', most_good)
'''Counter gives a list of tuples with word, couter'''



texts_3 = [row[access['Review Text']] for row in data if row[access['Rating']] < 4]
#5278 are bad reviews
texts_bad = ' '.join(texts_3)
#Remove Punctuations
word_review_bad = re.sub('[^A-Za-z]+', ' ', texts_bad)
word_tokens_bad = word_tokenize(word_review_bad)
filtered_word_bad = [w for w in word_tokens_bad if not w in nltk_words and len(w) > 2]
most_bad = Counter(filtered_word_bad).most_common(top_N)
print('Words which are often used in negative reviews are', most_bad)


###Plot the result

###Polarity

bloblist_desc = list()

review_text = [row[access['Review Text']] for row in data]

for row in review_text:
    blob = TextBlob(row)
    bloblist_desc.append([row, blob.sentiment.polarity, blob.sentiment.subjectivity])
#df_polarity_desc = pd.DataFrame(bloblist_desc, columns = ['Review','sentiment','polarity'])
 
def sentiment(pol_val):
    if pol_val > 0:
        val = "Positive Review"
    elif pol_val == 0:
        val = "Neutral Review"
    else:
        val = "Negative Review"
    return val

for row in bloblist_desc:
	row[1] = sentiment(row[1])


df_polarity_desc['Sentiment_Type'] = df_polarity_desc.apply(f, axis=1)

##Review length
review_length = [len(row[access['Review Text']]) for row in data]

