from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import string
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

app = Flask(__name__)

# Load spacy model
nlp = spacy.load('en_core_web_sm')

# Load dataset
df = pd.read_csv('imdb_labelled.txt',sep='\t', header=None)
df.columns = ['Reviews', 'Sentiment']

# Preprocess data
def text_data_cleaning(sentence):
    doc = nlp(sentence)
    tokens = []
    for token in doc:
        if token.lemma_ != "-PRON-":
            temp = token.lemma_.lower().strip()
        else:
            temp = token.lower_
        tokens.append(temp)
    cleaned_tokens = []
    for token in tokens:
        if token not in stopwords and token not in punct:
            cleaned_tokens.append(token)
    return cleaned_tokens

# Define stopwords and punctuations
punct = string.punctuation
stopwords = list(STOP_WORDS)

# Split data and train classifier
x = df['Reviews']
y = df['Sentiment']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
tfidf = TfidfVectorizer(tokenizer=text_data_cleaning)
clf = Pipeline([('tfidf', tfidf), ('clf', LinearSVC())])
clf.fit(x_train, y_train)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form['review']
        prediction = clf.predict([user_input])
        sentiment = "Positive" if prediction[0] == 1 else "Negative"
        return jsonify({"sentiment": sentiment})
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
