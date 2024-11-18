import os
import re
import PyPDF2
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import pickle
from nltk.stem.snowball import SnowballStemmer

class DocumentClassification:
    def __init__(self, data_directory):
        self.data_directory = data_directory
        self.clf = None
        self.vectorizer = None
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = SnowballStemmer('english')

    def read_text_files(self, directory):
        documents = []
        labels = []
        for category in os.listdir(directory):
            category_dir = os.path.join(directory, category)
            for file_name in os.listdir(category_dir):
                file_path = os.path.join(category_dir, file_name)
                with open(file_path, 'r') as f:
                    text = f.read()
                documents.append(text)
                labels.append(category)
        return documents, labels

    def preprocess_text(self, text):
        # convert to lowercase
        text = text.lower()

        # remove non-alphanumeric characters
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

        # tokenize
        tokens = text.split()

        # remove stopwords, single-letter words, and stem
        processed_tokens = []
        for token in tokens:
            if token not in self.stop_words and len(token) > 1:
                stemmed_token = self.stemmer.stem(token)
                processed_tokens.append(stemmed_token)

        # join processed tokens
        processed_text = ' '.join(processed_tokens)
        return processed_text


    def train_naive_bayes_classifier(self):
        documents, labels = self.read_text_files(self.data_directory)
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(documents)
        X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
        clf = MultinomialNB()
        clf.fit(X_train, y_train)
        self.clf = clf
        self.vectorizer = vectorizer
        
        # save model to disk
        with open('model.pkl', 'wb') as f:
            pickle.dump((self.clf, self.vectorizer), f)

    def classify_pdf_file(self, file_path):
        text = self.extract_text_from_pdf(file_path)
        processed_text = self.preprocess_text(text)
        X = self.vectorizer.transform([processed_text])
        y_pred = self.clf.predict(X)[0]
        return text,processed_text,y_pred

    def extract_text_from_pdf(self, file_path):
        # importing required modules
        from PyPDF2 import PdfReader
        filePath = file_path
        # creating a pdf reader objectf
        reader = PdfReader(filePath)
        
        # printing number of pages in pdf file
        # print(len(reader.pages))
        
        text=""
        # getting a specific page from the pdf file
        for page in reader.pages:
            text = text +" "+page.extract_text()
        # print(text)
        return text

    def evaluate_model(self):
        documents, labels = self.read_text_files(self.data_directory)
        X = self.vectorizer.transform(documents)
        y_true = labels
        y_pred = self.clf.predict(X)
        cm = confusion_matrix(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        acc = float(acc) * 100
        return cm, acc
    
    def predict(self, file_path):
        with open('model.pkl', 'rb') as f:
            self.clf, self.vectorizer = pickle.load(f)
        text = self.extract_text_from_pdf(file_path)
        processed_text = self.preprocess_text(text)
        X = self.vectorizer.transform([processed_text])
        y_pred = self.clf.predict(X)[0]
        return y_pred
