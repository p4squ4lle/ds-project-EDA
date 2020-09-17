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


houses.dropna(how='any', subset=['view', 'waterfront'], inplace=True)


houses.eval("price_per_sqft = price / sqft_living", inplace=True)
houses.head()


#renovated = []
#for value in houses.yr_renovated.values:
#    if value > 0:
#        renovated.append(1)
#    else:
#        renovated.append(0)
#
#houses["renovated"] = renovated


X_dum=houses.copy()
columns_to_drop = ["id", "date", "zipcode", "renovated", "price_per_sqft"]
for col in columns_to_drop:
    X_dum.drop(col, axis=1, inplace=True)

X_dum.columns


mask = np.triu(X_dum.corr())

fig, ax = plt.subplots(figsize=(15,10))
ax = sns.heatmap(round(X_dum.corr(),2)
                 ,annot=True
                 ,mask=mask
                 ,cmap='coolwarm')


#variables = ["sqft_living", "bathrooms", "bedrooms", "grade", "sqft_above", "sqft_living15", "view"]

variables = ["sqft_living", "grade", "yr_built", "view"]


train, test = train_test_split(houses, train_size = 0.8)
model = linear_model.LinearRegression()
model.fit(train[variables], train["price"])


print(model.score(train[variables], train['price']),
    model.score(test[variables], test['price']))


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))


predicted_values = model.predict(test[variables])
mape(test["price"], predicted_values)


import statsmodels.formula.api as smf


model = 'price ~ sqft_living + grade + view + yr_built'
result = smf.ols(formula=model, data=houses).fit()


result.summary()



