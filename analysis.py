'''
The Module can be used to read file and analyse the data. It has thr parts:
1) Read csv file, convert data into list, such that each row is a list and there is a dictionary which allow
access to different columns.
2) Data imputation: This involves changing strings to integers (list converts all objects into strings)
imputing no values to some value.
3) Summarizing the quantitative and qualitative variables. Provides, mean, median, max and min for quantitative
data and mode, categories for qualitative variables.
4) Print the summaries.
'''

import csv
from collections import Counter
import statistics as st
import re


csv_file = open('data.csv', encoding = "utf8", newline = '')
data_csv = csv.reader(csv_file, delimiter=',')


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

# We can use col_name_cat = ['Division Name', 'Department Name', 'Class Name']	
def summary_of_qualitative(data, access, column):
	'''the function returns unique set of categories and the counts of that categories.'''
	list_of_column = [row[access[column]] for row in data]
	column_unique = set(list_of_column)
	value_cnt = Counter(list_of_column)
	print('Counts : {}'.format(len(list_of_column)))
	print('Unique categories are:')
	for i in column_unique:
		print(i)
	print('Mode of the categories is {}'.format(max(value_cnt)))


def summary_of_quantitative(data, access, column):
	'''function returns the summary of the quatitative data in the column specified.'''
	list_of_column = [row[access[column]] for row in data]
	counts = len(list_of_column)
	mean_col = st.mean(list_of_column)
	median_col = st.median(list_of_column)
	min_col = min(list_of_column)
	max_col = max(list_of_column)
	print('Counts : {:20}'.format(counts))
	print('Mean : {:22.2f}'.format(mean_col))
	print('Median : {:20.1f}'.format(median_col))
	print('Minimum :{:20}'.format(min_col))
	print('Maximum :{:20}'.format(max_col))

def print_summary(data, access, col_list):
	'''something to add....'''
	col_name = ', '.join(col_list) #row 23
	print("Column names are {}".format(col_name[1:]))
	print("Processed {} lines.".format(len(data)))
	print('The following are the columns with type:')
	for key in access:
		print('{0:35} type is : {1}'.format(key, type(data[0][access[key]])))



def impute_data(data, access, column, from_value, to_value):
	'''function imputes the data accoding from the value to a given value. For example if we want to 
	impute the quantitative values with mean or median.'''
	for i, row in enumerate(data):
		if row[access[column]] == from_value:
			row[access[column]] = to_value

### Analyse the data:
(data, access, col_list) = convert_csv_into_list(data_csv)
def analyse_data():
	'''Analysis part.'''
	print_summary(data, access, col_list)
	print('')
	print('All columns are in string format. We have to convert some columns to integer format.')
	convert_string_to_integer(data, access)
	print_summary(data, access, col_list)
	print('')
	print('Now the columns Age, Positive Feedback Count and Rating are integers.')
	col_list2 = ['Rating', 'Positive Feedback Count', 'Age']
	for column in col_list2:
		print('The summary of {} is:'.format(column))
		summary_of_quantitative(data, access, column)
	


analyse_data()
col_list3 = ['Division Name', 'Department Name', 'Class Name']
for i in col_list3:
	impute_data(data, access, i, '', 'Others')
for i in col_list3:
	print('The summary of {} is:'.format(i))
	summary_of_qualitative(data, access, i)

csv_file.close()

