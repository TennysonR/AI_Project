#import Flask
from flask import Flask, render_template, request
from textblob import TextBlob
import numpy as np
import pandas as pd
import pickle
import re 

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict/', methods=['GET','POST'])
def predict():
    if request.method == "POST":
        #get form data
        words = request.form.get('words')
        likes = request.form.get('likes')
        time = request.form.get('time')

        data = {'tweets':[words], 'likes':[likes], 'time':[time]}
        #call preprocessDataAndPredict and Pass inputs
        prediction = preprocessDataAndPredict(data)
        #pass prediction to template
        return render_template('predict.html', prediction = prediction)
    pass

def preprocessDataAndPredict(data):
    test_data = pd.DataFrame(data)    

    #create a function to clean the texts
    import re
    def cleanTxt(text):
        text=re.sub(r'@[A-Za-z0-9]+ ,','',text)#removed @mentions
        text=re.sub(r'#','',text) #removing the #symbol
        text=re.sub(r'RT[\s]+','',text)#removing RT
        text=re.sub(r'https?:\/\/\S+','',text)#remove the hyper link

        return text

    # cleaning the tweets
    test_data['tweets']=test_data['tweets'].apply(cleanTxt)

    #create a function to get the subjectivity
    def getSubjectivity(text):
        return TextBlob(text).sentiment.subjectivity

    #create a function to get the polarity
    def getPolarity(text) :
        return     TextBlob(text).sentiment.polarity

    #create two new columns
    test_data['Subjectivity']=test_data['tweets'].apply(getSubjectivity)
    test_data['Polarity']=test_data['tweets'].apply(getPolarity)

    #create a function to compute the negative, positive and neutral sentiments

    test_data['tweets'],_=pd.factorize(test_data['tweets'])
    test_data['likes'],_=pd.factorize(test_data['likes'])
    test_data['time'],_=pd.factorize(test_data['time'])
    test_data['Polarity'],_=pd.factorize(test_data['Polarity'])

    with open('DecisionTree.plk', 'rb') as file:  
        trained_model = pickle.load(file)

    prediction = trained_model.predict(test_data)    
    return prediction

    
    pass
pass
if __name__ == '__main__':
    app.run(debug=True)

