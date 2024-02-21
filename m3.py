import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


store_sales=pd.read_csv("output2.csv")
store_sales = store_sales[['purchase date', 'quantity']]
store_sales['purchase date'] = pd.to_datetime(store_sales['purchase date'])

montly_sales = store_sales.groupby(store_sales['purchase date'].dt.to_period("M"))['quantity'].sum().reset_index()

montly_sales['purchase date']=montly_sales['purchase date'].dt.to_timestamp()
montly_sales['sales_diff']=montly_sales['quantity'].diff()
montly_sales=montly_sales.dropna()
supervised_data=montly_sales.drop(['purchase date','quantity'],axis=1)
for i in range(1,13):
  col_name='month'+str(i)
  supervised_data[col_name]=supervised_data['sales_diff'].shift(i)
supervised_data=supervised_data.dropna().reset_index(drop=True)
supervised_data.head(10)
train_data=supervised_data[:-12]
test_data=supervised_data[-12:]
print("train data shape: ", train_data.shape)
print("test data shape: ", test_data.shape)
scaler = MinMaxScaler(feature_range=(-1,1))
scaler.fit(train_data)
train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)
x_train,y_train=train_data[:,1:],train_data[:,0:1]
x_test,y_test=test_data[:,1:],test_data[:,0:1]
y_train=y_train.ravel()
y_test=y_test.ravel()
print("x_train shape:",x_train.shape)
print("y_train shape:",y_train.shape)
print("x_test shape:",x_test.shape)
print("y_test shape:",y_test.shape)
sales_dates=montly_sales['purchase date'][-12:].reset_index(drop=True)
predict_df=pd.DataFrame(sales_dates)
act_sales=montly_sales['quantity'][-13:].to_list()
print(act_sales)
lr_model=LinearRegression()
lr_model.fit(x_train,y_train)
lr_pre=lr_model.predict(x_test)
lr_pre=lr_pre.reshape(-1,1)
lr_pre_test_set=np.concatenate([lr_pre,x_test],axis=1)
lr_pre_test_set =  scaler.inverse_transform(lr_pre_test_set)
from operator import index
result_list=[]
for index in range(0,len(lr_pre_test_set)):
  result_list.append(lr_pre_test_set[index][0]+act_sales[index])
lr_pre_series = pd.Series(result_list, name="Linear Prediction")
predict_df = predict_df.merge(lr_pre_series, left_index=True, right_index=True)
lr_mse = np.sqrt(mean_squared_error(predict_df['Linear Prediction'],montly_sales['quantity'][-12:]))
lr_mae = mean_absolute_error(predict_df['Linear Prediction'],montly_sales['quantity'][-12:])
lr_r2 = r2_score(predict_df['Linear Prediction'],montly_sales['quantity'][-12:])

print("Linear Regression MSE: " ,lr_mse)
print("Linear Regression MAE: " ,lr_mae)
print("Linear Regression R2: " ,lr_r2)
plt.figure(figsize=(15,5))
plt.plot(montly_sales['purchase date'],montly_sales['quantity'])
plt.plot(predict_df['purchase date'],predict_df['Linear Prediction'])
plt.title("Customer sales forecast")
plt.xlabel("dates")
plt.ylabel("sales")
plt.legend(['actual sales','predict sales'])
plt.savefig("output_graph.png")