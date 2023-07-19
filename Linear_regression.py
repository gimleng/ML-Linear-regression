import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns

plt.rc("font", size=5)

### Import water use data to Pandas Dataframe
df = pd.read_csv('data/bn_water_use.csv', encoding='utf-8')
# remove features that not impact to analysis
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df.drop(['USENAME'], axis=1)
df = df.drop(['PRESENT_METER_DATE'], axis=1)

### Data exploration
df.info()
print(df.describe())

### Find correlation between parameters
df_corr = df.copy()
print(df_corr.corr())
# sns.heatmap(df_corr.corr(),annot=True)
# result_img: /Linear_regression_result_img/1_params_correlation.png
# result: No feature that we can use in linear regression method

### Solution: 1 --> Join data with another file to get more features data
df = df.sort_values(by=['CUST_CODE', 'DEBT_YM'])
df2 = pd.read_csv('data/data_meter_all.csv', encoding='utf-8')
# remove features that not impact to analysis
df2 = df2.drop(['regis_no'], axis=1)
df2 = df2.drop(['meterno'], axis=1)
df2 = df2.drop(['custname'], axis=1)
df2 = df2.drop(['custaddr'], axis=1)
df2 = df2.drop(['custpost'], axis=1)
df2 = df2.drop(['custtel'], axis=1)
df2 = df2.drop(['nearlocate'], axis=1)
df2 = df2.drop(['metermake'], axis=1)
df2 = df2.drop(['bgncustdt_r'], axis=1)
df2 = df2.drop(['bgnmtrdt_r'], axis=1)
df2 = df2.drop(['lstmtrddt_r'], axis=1)
df2 = df2.drop(['update_date'], axis=1)
df2 = df2.drop(['wkb_geometry'], axis=1)
df2 = df2.drop(['is_customer'], axis=1)
df2 = df2.drop(['a_code'], axis=1)
df2 = df2.drop(['tel'], axis=1)
df2 = df2.drop(['mobile'], axis=1)
df2 = df2.drop(['meterstat'], axis=1)
df2 = df2.drop(['t_code'], axis=1)
df2 = df2.drop(['p_code'], axis=1)
df2 = df2.drop(['custstat'], axis=1)
df2 = df2.drop(['ba'], axis=1)
df2 = df2.drop(['mtrmkcode'], axis=1)
df2 = df2.rename(columns = {'custcode':'CUST_CODE'})
df2 = df2.rename(columns = {'usetype':'use_type'})
df_join_1 = df.join(df2.set_index('CUST_CODE'), on='CUST_CODE' )
# sns.heatmap(df_join_1.corr(),annot=True)
# result_img: /Linear_regression_result_img/2_params_correlation.png
# result: use PRESENT_METER_COUNT as a main row to do logictic regression 
# select features that has high correlation with PRESENT_METER_COUNT to do the next step

df_selected_1 = pd.DataFrame()
df_selected_1 = pd.concat([df_selected_1,df2['CUST_CODE']], axis = 1)
df_selected_1 = pd.concat([df_selected_1,df2['metersize']], axis = 1)
df_selected_1 = pd.concat([df_selected_1,df2['prsmtrcnt']], axis = 1)
df_selected_1 = pd.concat([df_selected_1,df2['avgwtusg']], axis = 1)
df_selected_1 = pd.concat([df_selected_1,df2['all_wtr']], axis = 1)
df_selected_1 = pd.concat([df_selected_1,df2['avg_before12']], axis = 1)
df_join_2 = df.join(df_selected_1.set_index('CUST_CODE'), on='CUST_CODE' )
# sns.heatmap(df_join_2.corr(),annot=True)
# result_img: /Linear_regression_result_img/3_params_correlation.png

### remove unnecessary features from Dataframe
df_join_2 = df_join_2.drop(['CUST_ID'], axis=1)
df_join_2 = df_join_2.drop(['CUST_CODE'], axis=1)
df_join_2 = df_join_2.drop(['DEBT_YM'], axis=1)
df_join_2 = df_join_2.drop(['USETYPE'], axis=1)
# sns.heatmap(df_join_2.corr(),annot=True)
# result_img: /Linear_regression_result_img/4_params_correlation.png

### Visualizations
# df_join_2.hist(bins=50, figsize=(20,15))
# plt.savefig("attribute_histogram_plots")
# result_img: /Linear_regression_result_img/1_histogram_plots.png
# sns.pairplot(df_join_2)

### Data Preprocessing
df_join_3 = df.join(df_selected_1.set_index('CUST_CODE'), on='CUST_CODE' )
df_join_3 = df_join_3.drop(['CUST_ID'], axis=1)
df_join_3 = df_join_3.drop(['CUST_CODE'], axis=1)
df_join_3 = df_join_3.drop(['DEBT_YM'], axis=1)
df_join_3 = df_join_3.drop(['USETYPE'], axis=1)
df_join_3 = df_join_3.fillna(0) # replace NaN values with 0

### Define Train and Test datasets
X = df_join_3.loc[:, df_join_3.columns != 'PRESENT_WATER_USG']
y = df_join_3.loc[:, df_join_3.columns == 'PRESENT_WATER_USG']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
print("Total data: ",len(X))
print("Train data: ",len(X_train))
print("Test data: ",len(X_test))

### Features selection by Recursive Feature Elimination (RFE)
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
df_join_3_data = df_join_3.columns.values.tolist()
y1 = ['y']
X1 = [i for i in df_join_3_data if i not in y1]

estimator=LinearRegression()
rfe = RFE(estimator, n_features_to_select=7)
rfe = rfe.fit(X_train, y_train.values.ravel())
print(rfe.support_)
print(rfe.ranking_)

### Implementing model
import statsmodels.api as st
X1 = st.add_constant(X)
lin_model = st.OLS(y,X1)
result = lin_model.fit()
print(result.summary2())
# result_img: /Linear_regression_result_img/statsmodels.png
# result: All p-value is smaller than 0.05, except two variables, so i should be remove

### remove features that p-value greater than 0.05, and do the same method
df_join_4 = df_join_3.copy()
df_join_4 = df_join_4.drop(['all_wtr'], axis=1)
df_join_4 = df_join_4.drop(['avg_before12'], axis=1)
X_v2 = df_join_4.loc[:, df_join_4.columns != 'PRESENT_WATER_USG']
y_v2 = df_join_4.loc[:, df_join_4.columns == 'PRESENT_WATER_USG']

X_train_v2, X_test_v2, y_train_v2, y_test_v2 = train_test_split(X_v2, y_v2, test_size=0.30, random_state=42)

df_join_4_data = df_join_4.columns.values.tolist()
y1_v2 = ['y']
X1_v2 = [i for i in df_join_4_data if i not in y1_v2]

rfe2 = rfe.fit(X_train_v2, y_train_v2.values.ravel())
print(rfe2.support_)
print(rfe2.ranking_)

X1_v2 = st.add_constant(X_v2)
lin_model_2 = st.OLS(y_v2,X1_v2)
result_2 = lin_model_2.fit()
print(result_2.summary2())
# result_img: /Linear_regression_result_img/statsmodels_2.png
# result: All p-value is smaller than 0.05, so we can use this features to do model in the next step

### Linear Regression Model Fitting
from sklearn.linear_model import LinearRegression
from sklearn import metrics
estimator.fit(X_train_v2, y_train_v2)

### Predicting the test sets result and calculating precision
y_pred = estimator.predict(X_test_v2)
from sklearn.metrics import r2_score
print("Linear Regression R squared of the test set is: {:.2f}".format(r2_score(y_test_v2, y_pred)))

from sklearn.metrics import mean_squared_error
print("Linear Regression MSE of the test set is: {:.2f}".format(mean_squared_error(y_test_v2, y_pred)))

lin_rmse = np.sqrt(mean_squared_error(y_pred, y_test_v2))
print("Linear Regression RMSE of the test set is: {:.2f}".format(lin_rmse))

from sklearn.metrics import mean_absolute_error
print("Linear Regression MAE of the test set is: {:.2f}".format(mean_absolute_error(y_test_v2, y_pred)))

### Compare between predict value and true value 
y_test_v2_np = y_test_v2.to_numpy()
d = {'true': y_test_v2_np.flatten(), 'predicted': y_pred.flatten()}
df_comparison = pd.DataFrame(data=d)
df_comparison['diff'] = df_comparison['predicted'] - df_comparison['true']
# print(df_comparison.head())

print("length of the test data is ",len(X_test))
print("number of the overestimations for the true value is ",len(df_comparison['diff'][df_comparison['diff']>0]))
print("number of the underestimations for the true value is ",len(df_comparison['diff'][df_comparison['diff']<0]))
print("number of the exact estimations for the true value is ",len(df_comparison['diff'][df_comparison['diff']==0]))

sns.set(style="white")
plt.hist(df_comparison['diff'], bins=25, color="pink", edgecolor='brown', linewidth=2)
plt.axvline(0, color="red", linestyle='dashed', linewidth=2)
# plt.show()
# result_img: /Linear_regression_result_img/result_histogram.png
# result: There is a tendency of overestimation for the true test values

df_forecast = df_comparison.copy()
df_forecast['10_percentage_of_true'] = (df_forecast['true'] * 0.1)
# print("length of the test data is ",len(X_test))
# print("number of predict values is more than 10 percentage of true value: ",len(df_forecast['diff'][df_forecast['diff']>df_forecast['10_per']]))
# print("number of predict values is less than 10 percentage of true value: ",len(df_forecast['diff'][df_forecast['diff']<df_forecast['10_per']]))
# print("number of predict values is in 10 percentage of true value: ",len(X_test) - abs(len(df_forecast['diff'][df_forecast['diff']>df_forecast['10_per']]) + len(df_forecast['diff'][df_forecast['diff']<df_forecast['10_per']])))
df_forecast.to_csv("output/df_forecast.csv")