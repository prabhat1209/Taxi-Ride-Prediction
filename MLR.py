import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv('taxi.csv')
#print(data.head())

data_x = data.iloc[:,0:-1].values
data_y = data.iloc[:,-1].values

#print(data_x)
#print(data_y)

train_x,test_x,train_y,test_y = train_test_split(data_x,data_y,test_size=0.3,random_state=0)

reg = LinearRegression()
reg.fit(train_x,train_y)

pickle.dump(reg,open('taxi.pkl','wb'))

model = pickle.load(open('taxi.pkl','rb'))
print(model.predict([[80,1770000,6000,85]]))
