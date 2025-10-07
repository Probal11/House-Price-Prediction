import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
data=pd.read_csv('house_prices_dataset.csv')
X=data.drop('price',axis=1)
y=data.price
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
model=LinearRegression()
model.fit(X_train,y_train)
feet=float(input("Enter the total area in square feet of your house:"))
rm=int(input("How many rooms are present in your house:"))
ager=int(input("How old is your house(in years):"))
dist=float(input("How far is your house from the city(in termms of km):"))
arr=pd.DataFrame([feet,rm,ager,dist])
print("Now performing prediction...")
prediction=model.predict([[feet,rm,ager,dist]])
for i in prediction:
    print(f"The price of the house would be: Rs{i}")
print(f'The accuracy of the model is: {model.score(X_test,y_test)*100}%')