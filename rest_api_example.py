from flask import Flask, request, jsonify # loading in Flask
import pandas as pd # loading pandas for reading csv
import random
import numpy as np
from sklearn.model_selection import train_test_split
data=pd.read_csv('cpdata.csv')
pp = {'wheat': 0,'mungbean': 1,'Tea':2,'millet':3,'maize':4,'lentil':5,'jute':6,'cofee':7,'cotton':8,'ground nut':9,'peas':10,'rubber':11,'sugarcane':12,'tobacco':13,'kidney beans':14,'moth beans':15,'coconut':16,'blackgram':17,'adzuki beans':18,'pigeon peas':19,'chick peas':20,'banana':21,'grapes':22,'apple':23,'mango':24,'muskmelon':25,'orange':26,'papaya':27,'watermelon':28,'pomegranate':29}      
#data.previouscrop = [pp[item] for item in data.previouscrop]
label= pd.get_dummies(data.label).iloc[: , 1:]
data= pd.concat([data,label],axis=1)
data.drop('label', axis=1,inplace=True)
print('The data present in one row of the dataset is')
print(data.head(1))
train=data.iloc[:, 0:5].values
test=data.iloc[: ,5:].values

#Dividing the data into training and test set
X_train,X_test,y_train,y_test=train_test_split(train,test,test_size=0.3)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Importing Decision Tree classifier
from sklearn.tree import DecisionTreeRegressor
clf=DecisionTreeRegressor()

#Fitting the classifier into training set
clf.fit(X_train,y_train)
pred=clf.predict(X_test)

from sklearn.metrics import accuracy_score
# Finding the accuracy of the model
a=accuracy_score(y_test,pred)
print("The accuracy of this model is: ", a*100)

#Using firebase to import data to be tested
from firebase import firebase
firebase =firebase.FirebaseApplication('https://croprotation-e9e6a-default-rtdb.firebaseio.com/')
tp=firebase.get('/Realtime',None)

ah=tp['Air Humidity']
atemp=tp['Air Temp']
shum=tp['Soil Humidity']
pH=tp['Soil pH']
rain=tp['Rainfall']
prev=20

l=[]
l.append(ah)
l.append(atemp)
l.append(pH)
l.append(rain)
l.append(prev)

predictcrop=[l]

# Putting the names of crop in a single list
crops=['wheat','mungbean','Tea','millet','maize','lentil','jute','cofee','cotton','ground nut','peas','rubber','sugarcane','tobacco','kidney beans','moth beans','coconut','blackgram','adzuki beans','pigeon peas','chick peas','banana','grapes','apple','mango','muskmelon','orange','papaya','watermelon','pomegranate']
cr='rice'

#Predicting the crop
predictions = clf.predict(predictcrop)
count=0
for i in range(0,30):
    if(predictions[0][i]==1):
        c=crops[i]
        count=count+1
        break;
    i=i+1
if(count==0):
    print('The predicted crop is %s'%cr)
else:
    print('The predicted crop is %s'%c)

#Sending the predicted crop to database
cp=firebase.put('/croppredicted','crop',c)

# creating a Flask application
app = Flask(__name__)

# Load the model
model = clf

# creating predict url and only allowing post requests.
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from Post request
    data = request.get_json()
    # Make prediction
    df = pd.DataFrame([str(data['text'])], columns=['content'])
    print(df.head())
    # making predictions
    pred = model.predict(dataset=df, data_format='df')
    print(pred)
    # returning the predictions as json
    return jsonify(pred['airline_sentiment_predictions'][0])

if __name__ == '__main__':
    app.run(port=3000, debug=True)