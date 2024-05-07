from flask import Flask, request, jsonify
import re
import contractions
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import wordnet
import csv 
from spellchecker import SpellChecker
from langdetect import detect
import nltk
from nltk import pos_tag
from nltk.corpus import words
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy as sp
import scipy.sparse as sparse
from scipy.special import expit


nltk.download('wordnet')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')

class Sentiment_Analysis:
    
    # variables
    chat_words_dict = {}
    lemmatizer = WordNetLemmatizer()
    wordnet_map = {"N": wordnet.NOUN, "V": wordnet.VERB, "J": wordnet.ADJ, "R": wordnet.ADV}
    spellChecker = SpellChecker()
    englishWords = set(words.words())
    
    # methods
    def __init__(self) -> None:
        with open('abbr.txt') as f:
            for line in f:
                (key, val) = line.split(':', 1)
                self.chat_words_dict[key] = val.replace('\n', '')
    
    # Expand Abbreviations
    def convert_chat_words(self, text):
        words = text.split()
        converted_words = []
        for word in words:
            if word.lower() in self.chat_words_dict:
                converted_words.append(self.chat_words_dict[word.lower()])
            else:
                converted_words.append(word)
        converted_text = " ".join(converted_words)
        return converted_text

    # Lemmatization
    # Function to perform Lemmatization on a text
    def lemmatize_text(self, text):
        # Get the POS tags for the words
        pos_tags = nltk.pos_tag(text)
        
        # Perform Lemmatization
        lemmatized_words = []
        for word, tag in pos_tags:
            # Map the POS tag to WordNet POS tag
            pos = self.wordnet_map.get(tag[0].upper(), wordnet.NOUN)
            # Lemmatize the word with the appropriate POS tag
            lemmatized_word = self.lemmatizer.lemmatize(word, pos=pos)
            # Add the lemmatized word to the list
            lemmatized_words.append(lemmatized_word)
        
        return lemmatized_words

    # Remove Numbers 
    def remove_numbers(self, text):
        words = list()
        for word in text:
            if  word.isalpha():
                words.append(word)
        return words

    def preprocessTweet(self, text):
        #Lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove mentions
        text = re.sub(r'@\S+', '', text)
        
        # Remove hashtags
        text = re.sub(r'#\S+', '', text)
        
        # Expand contractions
        text = contractions.fix(text)
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        
        # Convert chat words
        text = self.convert_chat_words(text)
        
        # Tokenize text
        tokens = text.split()
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        
        # Lemmatize text
        tokens = self.lemmatize_text(tokens)

        # Remove numbers
        tokens = self.remove_numbers(tokens)

        return tokens

    # Remove Non-English Words & Spell Check
    def isEnglish(self, word):
        return word.lower() in self.englishWords

    def cleanTweet(self, tweet):
        cleaned_tokens = self.preprocessTweet(tweet)
        cleaned_batch = []
        for word in cleaned_tokens:
            if not self.isEnglish(word):
                continue

            if word.lower() not in self.spellChecker:
                continue

            cleaned_batch.append(word)
        return cleaned_batch

    def convertTFIDF(self, preprocessed_tweet):
        with open('dictionaryUnique.txt', 'r') as f:
            dictionary = f.read().splitlines()

        cleanedTokens = ' '.join(preprocessed_tweet)
        lastRow = pd.DataFrame({'tokens': [cleanedTokens]})
        f_df = pd.read_csv('f.csv')
        f_df = pd.concat([f_df, lastRow], ignore_index=True)
        f_df.to_csv('f.csv', index=False)

        vectorizer = TfidfVectorizer(vocabulary=dictionary)
        tfidf_vectors = vectorizer.fit_transform(f_df['tokens'].values.astype('U'))

        lastIndex = f_df.shape[0] - 1
        tfidfVector = tfidf_vectors[lastIndex]

        return tfidfVector
    
    def loadWeights(self):
        weights = np.load('weights.npy')
        return weights

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict(self, tfidfVector, weights):  
        pred = np.dot(tfidfVector, weights.T)
        probability = self.sigmoid(pred)
        sentiment = 'Positive' if probability >= 0.5 else 'Negative'
        return sentiment
    
    def logisticRegression(x, y, learningRate=0.01, iterations=2000):
        features = x.shape[1]
        weights = np.zeros(features)

        for i in range(iterations):
            z = x.dot(weights)
            p = expit(z)

            gradient = x.T.dot(p - y)
            weights -= learningRate * gradient

        return weights
    
    def updateTfidf(self, tfidfVector, label):
        tfidf_vectors_updated = sparse.load_npz('tfidf.npz')
        text_df_updated = pd.read_csv('labels.csv')
        tfidf_vectors_updated = sparse.vstack([tfidf_vectors_updated, tfidfVector])
        new = pd.DataFrame({'target': [label]})
        text_df_updated = pd.concat([text_df_updated, new], ignore_index=True)

        sparse.save_npz('tfidf.npz', tfidf_vectors_updated)
        text_df_updated.to_csv('labels.csv', index=False)
        
    def retrainModel(self):
        tfidf_vectors_retrain = sparse.load_npz('tfidf.npz')
        text_df_retrain = pd.read_csv('labels.csv')
        x = tfidf_vectors_retrain
        y = text_df_retrain['target'].values
        weights = self.logisticRegression(x, y, learningRate=0.01, iterations=2000)
        np.save('weights.npy', weights)
    