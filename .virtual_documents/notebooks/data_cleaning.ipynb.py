reset -fs


import pandas as pd
import numpy as np
import seaborn as sns

import plotly.express as px
import matplotlib.pyplot as plt
get_ipython().run_line_magic("matplotlib", " widget")


houses = pd.read_csv("../data/King_County_House_prices_dataset.csv")


houses.head()


houses.shape


number_nans = {}

for col in houses.columns:
    number_nans[col] = 0
    for i, v in enumerate(houses[col].isnull()):
        if v:
            number_nans[col] += 1
            
number_nans
            


houses.info()


type(houses.sqft_basement[0])


try:
    houses.sqft_basement.astype(float)
except ValueError as e:
    print(e)


houses.sqft_basement.replace("?", np.nan, inplace=True)


houses.sqft_basement = houses.sqft_basement.astype(float)


houses.info()


dummy_series = houses.yr_renovated.isnull()

dummy_series = dummy_series.astype(int)

dummy_series.name = "never_renovated"


houses = pd.concat([houses, dummy_series], axis=1)


houses.head()


sns.boxplot(x="never_renovated", y="price", data=houses)



