import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('TSLA.csv') #reads data in csv file


#Converting string to datetime and changing to desired format
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True).dt.strftime('%m/%d/%Y')

#Close column and Adj column in the file share the same data
#df.drop removes redundant data in the dataset
df = df.drop(['Adj Close'], axis=1)
print(df.head()) #prints first five rows
print(df.shape) #provides number of rows and numebr of columns (2416, 7 or 6 after removing Adj Close column))
df.describe() #provides info of mean, std, min,count
df.isnull().sum() #Checks for null values in the dataset

#produces graph of closing price stocks from the Close column in the file
plt.figure(figsize=(15,5))
plt.plot(df['Close'])
plt.title('Tesla Close price.', fontsize=15)
plt.ylabel('Price in dollars.')
plt.show()


#Distribution plots visually assess the distribution of data comparing the empirical distribution of the data
# with the theoretical values expected from a specified distribution
#distribution plots
features = ['Open', 'High', 'Low', 'Close', 'Volume']
plt.subplots(figsize=(20, 10))
for i, col in enumerate(features):
  plt.subplot(2, 3, i + 1)
  sb.distplot(df[col])
plt.show()

#displays the five-number summary of a set of data.
#The five-number summary is the minimum, first quartile, median, third quartile, and maximum and any outliers.
#Outliers can reveal mistakes or unusual occurrences in data.
#box plots
plt.subplots(figsize=(20,10))
for i, col in enumerate(features):
  plt.subplot(2,3,i+1)
  sb.boxplot(df[col])
plt.show()
#volume data contains outliers in it


#Feature Egnineering
#helps to derive some valuable features from the existing ones. These extra features sometimes help
# in increasing the performance of the model significantly and certainly help to gain deeper insights into the data.
splitted = df['Date'].str.split('/', expand=True)
df['day'] = splitted[1].astype('int')
df['month'] = splitted[0].astype('int')
df['year'] = splitted[2].astype('int')
df['is_quarter_end'] = np.where(df['month']%3==0,1,0) #included as quarterly results impact stock prices,
# helpful feature for the learning model. 1 = yes 0 = no
print(df.head())

#df.groupby('is_quarter_end').mean() - should show difference between quarter end months
# Prices are higher in the months which are quarter end as compared to that of the non-quarter end months.
# The volume of trades is lower in the months which are quarter end.


#additional columns to help train the model
#Target feature is a signal whether to buy or not model trained to predict this only
df['open-close']  = df['Open'] - df['Close']
df['low-high']  = df['Low'] - df['High']
df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

plt.pie(df['target'].value_counts().values,
        labels=[0, 1], autopct='%1.1f%%')
plt.show()

#When we add features to our dataset we have to ensure that there are no highly correlated features
# as they do not help in the learning process of the algorithm.


features = df[['open-close', 'low-high', 'is_quarter_end']]
target = df['target']

scaler = StandardScaler()
features = scaler.fit_transform(features)

X_train, X_valid, Y_train, Y_valid = train_test_split(
    features, target, test_size=0.1, random_state=2022)
print(X_train.shape, X_valid.shape)

#selecting features to train the model and normalizing the data for stable and fast training of the model.
# After whole data has been split into two parts 90/10 ratio
# to evaluate the performance of our model on unseen data.

#ML model training

models = [LogisticRegression(), SVC(
    kernel='poly', probability=True), XGBClassifier()]

for i in range(3):
    models[i].fit(X_train, Y_train)

    print(f'{models[i]} : ')
    print('Training Accuracy : ', metrics.roc_auc_score(
        Y_train, models[i].predict_proba(X_train)[:, 1]))
    print('Validation Accuracy : ', metrics.roc_auc_score(
        Y_valid, models[i].predict_proba(X_valid)[:, 1]))
    print()

