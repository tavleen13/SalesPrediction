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


for item in all_items:
	item_df = pd.DataFrame(df_train.loc[df_train['Item_ID'] == item, ['Datetime', 'Item_ID', 'ID',
							'Category_1', 'Category_2', 'Category_3', 'Price', 'Number_Of_Sales']])

	item_df['Category_2'].fillna(mode(item_df['Category_2']).mode[0], inplace=True)
	item_df['Category_3'].fillna(item_df['Category_3'].mode())
	item_df['Category_1'].fillna(item_df['Category_1'].mean(), inplace=True)
	item_df['Price'].fillna(item_df['Price'].mean()
	frames.append(item_df)

	# ----- For plotting ------
	# temp = {}	
	# for b in bins:
	# 	temp[b]= item_df.loc[item_df['bin'] == b, 'Number_Of_Sales'].sum()
	# plt.plot(list(temp.keys()), list(temp.values()))

final_df = pd.concat(frames)

# Initialise regression model
regr = linear_model.LinearRegression()
regr_price = linear_model.LinearRegression()

train, validation = train_test_split(final_df, train_size=0.7)

train_x = train.ix[:,3:5]
train_y_price = train.ix[:,-2]
train_y_sales = train.ix[:,-1]

validate_x = validation.ix[:,2:4]
validate_y_price = validation.ix[:,-2]
validate_y_sales = validation.ix[:,-1]

test_df['Category_2'].fillna(mode(test_df['Category_2']).mode[0], inplace=True)
test_x = test_df.ix[:,2:4]

regr.fit(train_x, train_y_sales)

regr_price.fit(train_x, train_y_price)
predicted_sales = regr.predict(test_x)
predicted_price = regr_price.predict(test_x)
test_df['Number_Of_Sales'] = predicted_sales
test_df['Price'] = predicted_price

result = pd.DataFrame()
result['ID'] = test_df['ID']
result['Number_Of_Sales'] = test_df['Number_Of_Sales'].apply(int)
result['Price'] = test_df['Price']

result.to_csv('submit.csv', index=False)
print("Mean squared error: %.2f"
      % np.mean((regr.predict(validate_x) - validate_y_sales) ** 2))
print(item_df.head(20))
	time = df_train.loc[df_train['Item_ID'] == item, 'Datetime']








