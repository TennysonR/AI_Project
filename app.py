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

        data = {'tweets':[words], 'likes':[likes]}
        #call preprocessDataAndPredict and Pass inputs
        prediction = preprocessDataAndPredict(data)
        #pass prediction to template
        return render_template('predict.html', prediction = prediction)
    pass

def preprocessDataAndPredict(data):
    test_data = pd.DataFrame(data)    


    test_data['tweets'],_=pd.factorize(test_data['tweets'])
    test_data['likes'],_=pd.factorize(test_data['likes'])


    with open('DecisionTree.plk', 'rb') as file:  
        trained_model = pickle.load(file)

    prediction = trained_model.predict(test_data)    
    return prediction

    
    pass
pass
if __name__ == '__main__':
    app.run(debug=True)

