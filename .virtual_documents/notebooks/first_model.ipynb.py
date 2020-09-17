import pandas as pd
import numpy as np
import seaborn as sns

import plotly.express as px
import matplotlib.pyplot as plt
get_ipython().run_line_magic("matplotlib", " widget")


houses = pd.read_csv("../data/King_County_House_prices_dataset.csv")


columns_of_interest = list(houses.columns)

columns_of_interest = columns_of_interest[2:15] + columns_of_interest[19:]
print(columns_of_interest)


cont_columns = columns_of_interest[:5] + columns_of_interest[-5:]
print(cont_columns)

diff = lambda l1,l2: [x for x in l1 if x not in l2]
cat_columns = ["price"] + diff(columns_of_interest, cont_columns)
print(cat_columns)


sns.pairplot(houses[cont_columns]);


sns.pairplot(houses[cat_columns])


import statsmodels.api as sm
from itertools import combinations
import statsmodels.formula.api as smf


cont_columns


rs = {}

for combo in combinations(cont_columns[1:], 2):
    model = 'price ~ {} + {}'.format(combo[0],combo[1])
    rs[(combo[0], combo[1])] = smf.ols(formula=model, data=houses).fit().rsquared


for k, v in rs.items():
    if v > 0.5 and v get_ipython().getoutput("= 1:")
        print(k, v)


params = ["sqft_living", "bathrooms", "yr_built"]

model = "price ~ {} + {} + {}".format(*params)
result = smf.ols(formula=model, data=houses).fit()
result.summary()


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))


result.params


b0 = result.params[0]
b1 = result.params[1]
b2 = result.params[2]
b3 = result.params[3]

mape(houses.price, b0 + b1*houses[params[0]] + b2*houses[params[1]] + b3*houses[params[2]])






