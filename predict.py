import pandas as pd
import csv
import numpy
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from scipy.stats import mode
import numpy as np

df_train = pd.read_csv('/Users/tavleenkaur/Documents/fractal/train.csv')
test_df = pd.read_csv('/Users/tavleenkaur/Documents/fractal/test.csv')

all_items = df_train.Item_ID.unique()

#Put date in bins

# bin1 = 'Jun-2014'
# bin2 = 'Dec-2014'
# bin3 = 'Jun-2105'
# bin4 = 'Dec-2015'
# bin5 = 'Jun-2016'
# bin6 = 'Dec-2016'
# bins = [bin1, bin2, bin3, bin4, bin5, bin6]
# dates = df_train.Datetime.unique()
# # date_df = pd.DataFrame(dates)
# def binize_date(dates):
# 	temp_list = []
# 	for date in dates:
# 		year = datetime.strptime(date, '%Y-%m-%d').year
# 		month = datetime.strptime(date, '%Y-%m-%d').month
# 		if month >=1 and month <=6:
# 			if year == 2014:
# 				label = bin1
# 			elif year == 2015:
# 				label = bin3
# 			else:
# 				label = bin5
# 		else:
# 			if year == 2014:
# 				label = bin2
# 			elif year == 2015:
# 				label = bin4
# 			else:
# 				label = bin6
# 		temp_list.append([date, label])
# 	return temp_list

# #dates mapping to bin mapping
# date_df = pd.DataFrame(binize_date(dates), columns = ['Datetime', 'bin'])
# final_df = pd.DataFrame()
# frames = []
# test_frames = []
for item in all_items:
	item_df = pd.DataFrame(df_train.loc[df_train['Item_ID'] == item, ['Datetime', 'Item_ID', 'ID',
							'Category_1', 'Category_2', 'Category_3', 'Price', 'Number_Of_Sales']])
	# item_df_test = pd.DataFrame(df_train.loc[df_train['Item_ID'] == item, ['Item_ID', 'Datetime', 'ID', 
							# 'Category_1', 'Category_2', 'Category_3']])
	# item_df['ID'] = item_df.loc[item_df['Date']]
	# ''.join(s.split('-'))
	# x = date_df.loc[date_df['date'] == item_df['Datetime'], date_df['bin']]
	# item_df = pd.concat(item_df, )
	
	# item_df = item_df.merge(date_df,on='Datetime',
 #                   how='outer')
	# print(item_df.head(5))
	item_df['Category_2'].fillna(mode(item_df['Category_2']).mode[0], inplace=True)
	# item_df_test['Category_2'].fillna(mode(item_df_test['Category_2']).mode[0], inplace=True)

	item_df['Category_3'].fillna(item_df['Category_3'].mode())
	item_df['Category_1'].fillna(item_df['Category_1'].mean(), inplace=True)
	# item_df['Number_of_Sales'].fillna(item_df['Number_Of_Sales'].mean())
	item_df['Price'].fillna(item_df['Price'].mean())
	# item_df['Datetime'].apply(str)
	frames.append(item_df)
	# test_frames.append(item_df_test)
	# break

	# ----- For plotting ------
	# temp = {}	
	# for b in bins:
	# 	temp[b]= item_df.loc[item_df['bin'] == b, 'Number_Of_Sales'].sum()
	# 	print(b)
	# 	print(temp[b])
	# print(temp.keys())
	# print(temp.values())

	# plt.plot(list(temp.keys()), list(temp.values()))

final_df = pd.concat(frames)
test_df['Category_2'].fillna(mode(test_df['Category_2']).mode[0], inplace=True)
print(test_df.head(5))

regr = linear_model.LinearRegression()
regr_price = linear_model.LinearRegression()
train, validation = train_test_split(final_df, train_size=0.7)

train_x = train.ix[:,3:5]

train_y_price = train.ix[:,-2]
train_y_sales = train.ix[:,-1]

validate_x = validation.ix[:,2:4]
validate_y_price = validation.ix[:,-2]
validate_y_sales = validation.ix[:,-1]

test_x = test_df.ix[:,2:4]

regr.fit(train_x, train_y_sales)

regr_price.fit(train_x, train_y_price)
predicted_sales = regr.predict(test_x)
predicted_price = regr_price.predict(test_x)
test_df['Number_Of_Sales'] = predicted_sales
test_df['Price'] = predicted_price
print(test_df.head())
result = pd.DataFrame()
result['ID'] = test_df['ID']
result['Number_Of_Sales'] = test_df['Number_Of_Sales'].apply(int)
result['Price'] = test_df['Price']
print('\nResult\n')
print(result.head(5))
result.to_csv('submit.csv', index=False)
# print("Mean squared error: %.2f"
#       % np.mean((regr.predict(validate_x) - validate_y_sales) ** 2))
# print(item_df.head(20))
	# time = df_train.loc[df_train['Item_ID'] == item, 'Datetime']








