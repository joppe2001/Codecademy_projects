import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

df = pd.read_csv("honeyproduction.csv")


prod_per_year = df.groupby('year').totalprod.mean().reset_index()

print(prod_per_year) 

X = prod_per_year['year'].values.reshape(-1, 1)
y = prod_per_year['totalprod'].values.reshape(-1, 1)

regr = linear_model.LinearRegression()
regr.fit(X, y)

X_future = np.array(range(2013, 2050))
X_future = X_future.reshape(-1, 1)

print(regr.predict(X_future))

future_predict = regr.predict(X_future)

print(regr.coef_[0])
print(regr.intercept_[0])
    
y_pred = regr.predict(X)

plt.title("Honey Production")
plt.xlabel("Year")
plt.ylabel("Production")
plt.plot(X_future, future_predict, 'red')
plt.plot(X, y_pred)
plt.scatter(X, y)
plt.show()