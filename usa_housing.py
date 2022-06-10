#Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from sklearn.metrics import  mean_absolute_error
#Reading File
df = pd.read_csv('USA_Housing.csv')
temp = df.head()
#Data understanding
temp = df.describe() #Count, mean, std, min-max informations.
temp = df.info() #There are no not a number value in our dataset.
#Data visulation
sbn.distplot(df['Price']) #Our Y(Price)'s graphic is looking well. 
plt.show()
#Colleration Informations
colleration = df.corr()
temp = colleration['Price'].sort_values()
#Preparing to train process
#Firstly we have to drop 'Address column' because we dont need it.
df = df.drop('Address',axis=1)

y = df['Price'] #Price is a result we want to reach. 
x = df.drop('Price',axis=1) #Others is our features.

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=15)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

#Creating model
model = Sequential()
model.add(Dense(units=5,activation='relu')) #Input layer
model.add(Dense(units=3,activation='relu'))
model.add(Dense(units=3,activation='relu'))
model.add(Dense(units=1,activation='relu')) #Output layer
model.compile(optimizer='adam', loss='mse') #Compile

#Training
model.fit(x=x_train, y=y_train, epochs=100, validation_data=(x_test,y_test))
loss = pd.DataFrame(model.history.history)
loss.plot()
plt.show()

#Testing
house = df.drop('Price',axis=1).iloc[540]
house = scaler.transform(house)
temp = model.predict(house)

#print(temp)