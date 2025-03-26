from flask import Flask, render_template, request, jsonify
import tweepy
import joblib
import numpy as np
from nltk.stem import PorterStemmer

app = Flask(__name__)

modelo = joblib.load('../models/RNA5.pkl')
encoder = joblib.load('../models/encoder.pkl')
vectorizador = joblib.load('../models/vectorizador.pkl')
stemmer = PorterStemmer()

bearer = 'AAAAAAAAAAAAAAAAAAAAADmx0AEAAAAA40HeZnialu6TCsnc0sQY4PJIyZo%3DrvHkbehI6eg0QK97vuDff7Xdj4DsCbL72b0wv4X9XqMkK6m4zA'
client = tweepy.Client(bearer_token=bearer)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/buscar', methods=['POST'])
def buscar_tweets():
    query = request.form['query']
    tweets = client.search_recent_tweets(
        query=query,
        max_results=10,
        tweet_fields=["id", "text"]
    )
    
    tweet_data = []
    if tweets.data:
        for tweet in tweets.data:
            tweet_data.append({'id': tweet.id, 'text': tweet.text.replace('\n', ' ')})
    
    return jsonify(tweet_data)

@app.route('/analizar', methods=['POST'])
def analizar_tweets():
    tweets = request.json.get('tweets', [])
    resultados = []
    
    for tweet in tweets:
        vectorizado = vectorizador.transform([tweet['text']])
        prediccion = modelo.predict(vectorizado)
        clase_predicha = np.argmax(prediccion)
        etiqueta = encoder.inverse_transform([clase_predicha])[0]
        
        resultados.append({
            'id': tweet['id'],
            'text': tweet['text'],
            'categoria': etiqueta
        })
    
    return jsonify(resultados)

if __name__ == '__main__':
    app.run(debug=True)
