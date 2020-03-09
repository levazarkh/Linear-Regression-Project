import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


customers = pd.read_csv("Ecommerce Customers")

print(customers.head())
print(customers.describe())
print(customers.info())

sns.set_palette("GnBu_d")
sns.set_style('whitegrid')

# More time on site, more money spent.
sns.jointplot(x='Time on Website',y='Yearly Amount Spent',data=customers)
plt.show()

sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=customers)
plt.show()

sns.jointplot(x='Time on App',y='Length of Membership',kind='hex',data=customers)
plt.show()

sns.pairplot(customers)
plt.show()
# Length of Membership is the most correlated feature with Yearly Amount Spent

sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data=customers)
plt.show()

# training and testing data

# setting variable
y = customers['Yearly Amount Spent']
X = customers[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# training the model
from sklearn.linear_model import LinearRegression

# creating an instance of a linear regression model
lm = LinearRegression()

# training/fitting on the training data
lm.fit(X_train,y_train)

# The coefficients
print('Coefficients: \n', lm.coef_)

# predicting test data
predictions = lm.predict( X_test)

plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.show()

# evaluating the model
from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

# histogram of the residuals
sns.distplot((y_test-predictions),bins=50);
plt.show()

coeffecients = pd.DataFrame(lm.coef_,X.columns)
coeffecients.columns = ['Coeffecient']
print(coeffecients)

#
# Interpreting the coefficients:
#
# Holding all other features fixed, a 1 unit increase in Avg. Session Length is associated with an increase of 25.98 total dollars spent.
# Holding all other features fixed, a 1 unit increase in Time on App is associated with an increase of 38.59 total dollars spent.
# Holding all other features fixed, a 1 unit increase in Time on Website is associated with an increase of 0.19 total dollars spent.
# Holding all other features fixed, a 1 unit increase in Length of Membership is associated with an increase of 61.27 total dollars spent.
