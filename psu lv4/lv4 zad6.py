import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, max_error

df = pd.read_csv('cars_processed.csv')
print(df.info())

categorical_vars = ['fuel', 'seller_type', 'transmission', 'owner']

df_dummies = pd.get_dummies(df, columns=categorical_vars, drop_first=True)

# Ulazne varijable
X = df_dummies.drop('selling_price', axis=1)
y = df_dummies['selling_price']

# Numeričkih ulazne varijable
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
X_numeric = X[numeric_features]

# Podjela na train i test
X_train, X_test, y_train, y_test = train_test_split(X_numeric, y, test_size=0.2, random_state=300)

# Skaliranje numeričkih ulaznih varijabli
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)


linear_model = LinearRegression()
linear_model.fit(X_train_s, y_train)


y_pred_train = linear_model.predict(X_train_s)
y_pred_test = linear_model.predict(X_test_s)

print("R2 test", r2_score(y_pred_test, y_test))
print("RMSE test:", np.sqrt(mean_squared_error(y_pred_test, y_test)))
print("Max error test:", max_error(y_pred_test, y_test))
print("MAE test:", mean_absolute_error(y_pred_test, y_test))


fig = plt.figure(figsize=[13, 10])
ax = sns.regplot(x=y_pred_test, y=y_test, line_kws={'color': 'green'})
ax.set(xlabel='Predikcija', ylabel='Stvarna vrijednost', title='Rezultati na testnim podacima')
plt.show()

