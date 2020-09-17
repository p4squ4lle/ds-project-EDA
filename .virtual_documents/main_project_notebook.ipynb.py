import pandas as pd
import numpy as np
import seaborn as sns

import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics

import datetime
get_ipython().run_line_magic("matplotlib", " inline")


houses = pd.read_csv("data/King_County_House_prices_dataset.csv")

houses.head()


houses.shape


houses.dropna(how='any', subset=['view', 'waterfront'], inplace=True)


houses.shape


view_nan_index = []

for i, v in enumerate(houses["view"].isnull()):
    if v:
        view_nan_index.append(i)
            
len(view_nan_index)


houses.drop(view_nan_index, axis=0, inplace=True)
houses.shape


houses.describe()


water_nan_index = []

for i, v in enumerate(houses["waterfront"].isnull()):
    if v:
        water_nan_index.append(i)
         
#diff = lambda l1,l2: [x for x in l1 if x not in l2]
#water_nan_index = diff(water_nan_index, view_nan_index)
len(water_nan_index)


houses.drop(water_nan_index, axis=0, inplace=True)
houses.shape


cols = ['price_per_sqft', 'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_living15', 'yr_built']


sns.pairplot(houses[cols], kind="reg", 
             plot_kws={'line_kws':{'color':'red', 'alpha': 0.5}, 
                       'scatter_kws': {'alpha': 0.01}});


sns.boxplot(x=houses[cols[-1]].values, y=houses.price.values, data=houses)
plt.xlabel("grades");


sns.boxplot(x=houses.bedrooms.values, y=houses.price.values, data=houses)
plt.xlabel("bedrooms")


fig, ax = plt.subplots(figsize=(8,5))
sns.boxplot(x=np.ceil(houses.bathrooms.values/houses.bedrooms.values), y=houses.price_per_sqft.values, data=houses)
plt.xticks(rotation=70);
plt.xlabel("bathrooms per bedrooms", fontdict={"fontsize":14})
plt.ylabel("price per sqft / Dollar", labelpad=15, fontdict={"fontsize":14})
plt.xticks(fontsize=14)
plt.yticks(fontsize=14);


fig, ax = plt.subplots(figsize=(8,6))
sns.boxplot(x=houses.waterfront.values, y=houses.price_per_sqft.values, data=houses)
plt.xlabel("waterfront", fontdict={"fontsize":14})
plt.ylabel("price per sqft / Dollar", labelpad=15, fontdict={"fontsize":14})
plt.xticks(fontsize=14)
plt.yticks(fontsize=14);


sns.boxplot(x=houses.grade.values, y=houses.price_per_sqft.values, data=houses)
plt.xlabel("grades", fontdict={"fontsize":14})
plt.ylabel("price per sqft / Dollar", labelpad=15, fontdict={"fontsize":14})
plt.xticks(fontsize=14)
plt.yticks(fontsize=14);


fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(x=houses.view.values, y=houses.price.values, data=houses)
plt.xlabel("view", fontdict={"fontsize":14})
plt.ylabel("price per sqft / Dollar", labelpad=15, fontdict={"fontsize":14})
plt.xticks(fontsize=14)
plt.yticks(fontsize=14);


houses.eval("price_per_sqft = price / sqft_living", inplace=True)
houses.head()


#groupy = houses.groupby("sqft_lot").median()
#groupy.price_per_sqft.plot()
#sns.relplot(x=houses.sqft_lot.values, y=houses.price_per_sqft.values, data=houses, kind="line")
houses.columns


bins = [i for i in range(1900, 2023, 10)]
labels = [str(bins[i])+"-"+str(bins[i+1]) for i in range(len(bins)-1)]

yr_binned = pd.cut(houses.yr_built, bins=bins, labels=labels)
#houses_binned
yr_binned.name = "yr_binned"

houses_binned = pd.concat([houses, yr_binned], axis=1)
houses_binned.head()


dummyhouses = houses.copy()

dummyhouses.dropna(how='any', subset=['view', 'waterfront', 'yr_renovated'], inplace=True)


renovated = []
for value in dummyhouses.yr_renovated.values:
    if value > 0:
        renovated.append(1)
    else:
        renovated.append(0)

              


dummyhouses["renovated"] = renovated
#dummyhouses.head()
#houses.dropna(how='any', subset=['view', 'waterfront'], inplace=True)
#dummy_series = dummy_series.astype(int)
#
#dummy_series.name = "never_renovated"
#
#houses_binned = pd.concat([houses_binned, dummy_series], axis=1)
#
#houses_binned.head()


fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(x=dummyhouses.renovated.values, y=dummyhouses.price_per_sqft.values, data=dummyhouses)
plt.xlabel("renovated", fontdict={"fontsize":14})
plt.ylabel("price per sqft / Dollar", labelpad=15, fontdict={"fontsize":14})
plt.xticks(fontsize=14)
plt.yticks(fontsize=14);


grouped = houses_binned.groupby("yr_binned").median()

fig, ax = plt.subplots(figsize=(8,5))
sns.boxplot(x=houses_binned["yr_binned"].values, y=houses_binned.price_per_sqft.values, data=houses_binned)
plt.xlabel("year built", fontdict={"fontsize":14})
plt.ylabel("price per sqft / Dollar", labelpad=15, fontdict={"fontsize":14})
plt.xticks(fontsize=14, rotation=90)
plt.yticks(fontsize=14);


dates = houses.date.tolist()

dummyyy = houses.copy()
dummyyy['datetimes'] = [datetime.datetime.strptime(d,"get_ipython().run_line_magic("m/%d/%Y").date()", " for d in dates]")

dummyyy = dummyyy.groupby("datetimes").median()


dummyyy.plot(x="datetimes", y="price", data=dummyyy)

#x_values = [datetime.datetime.strptime(d,"get_ipython().run_line_magic("m/%d/%Y").date()", " for d in dates]")
#y_values = dummyyy.price.values

#plt.plot(x_values, y_values)


X_dum=houses.copy()
columns_to_drop = ["id", "date", "zipcode"]
for col in columns_to_drop:
    X_dum.drop(col, axis=1, inplace=True)

X_dum.columns


mask = np.triu(X_dum.corr())

fig, ax = plt.subplots(figsize=(15,10))
ax = sns.heatmap(round(X_dum.corr(),2)
                 ,annot=True
                 ,mask=mask
                 ,cmap='coolwarm')

#plt.savefig('figures/correlogram.png')


#variables = ["sqft_living", "bathrooms", "bedrooms", "grade", "sqft_above", "sqft_living15", "view"]

variables = ["sqft_living", "grade", "waterfront", "yr_built"]


train, test = train_test_split(houses, train_size = 0.8)
model = linear_model.LinearRegression()
model.fit(train[variables], train["price"])


model.intercept_


model.coef_


model.score(train[variables], train['price'])


model.score(test[variables], test['price'])


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))


predicted_values = model.predict(test[variables])


mape(test["price"], predicted_values)






