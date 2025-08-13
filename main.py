#import libraries
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

#get data from csv file
df= pd.read_csv('youtube_activewin_feed_labeled.csv')
df.info()

#drop unnecessary data from csv from active-win
def preprocessData(df):
    df.drop(columns=['platform','id','owner','bounds','memoryUsage'],inplace=True)
    return df

df=preprocessData(df)

#vectorize text
x_train,x_test,y_train,y_test=train_test_split(df['title'],df['label'],test_size=0.2,random_state=42)

vectorizer = TfidfVectorizer()
x_train_tfidf = vectorizer.fit_transform(x_train)
x_test_tfidf = vectorizer.transform(x_test)
#train model using logisitc regression
trainingModel=LogisticRegression()
trainingModel.fit(x_train_tfidf,y_train)

#test model accuracy
y_pred=trainingModel.predict(x_test_tfidf)
print("Score:"+ str(accuracy_score(y_test,y_pred)))

#singular test case 
new_site = ['The Manim Experience - Creating animations with Python']
new_site_tfidf=vectorizer.transform(new_site)


probability=trainingModel.predict_proba(new_site_tfidf)[0][1]
#display probability and result
print(probability)
if probability>0.7:
    print('Site is probably distracting')
else:
    print('This is educational!')