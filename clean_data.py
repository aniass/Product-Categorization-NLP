# Load libraries
import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

stop = stopwords.words('english')
porter = PorterStemmer()

URL_DATA = '\data\products_description.csv'


def remove_punctuation(description):
    """Function to remove punctuation"""
    table = str.maketrans('', '', string.punctuation)
    return description.translate(table)


def remove_stopwords(text):
    """Function to removing stopwords"""
    text = [word.lower() for word in text.split() if word.lower() not in stop]
    return " ".join(text)


def stemmer(stem_text):
    """Function to apply stemming"""
    stem_text = [porter.stem(word) for word in stem_text.split()]
    return " ".join(stem_text)


def read_data(path):
    """Function to read and clean text data"""
    data = pd.read_csv(path, header=0, index_col=0)
    data['description'] = data['description'].astype(str)
    data['description'] = data['description'].apply(remove_punctuation)
    data['description'] = data['description'].apply(remove_stopwords)
    data['description'] = data['description'].apply(stemmer)
    return data


if __name__ == '__main__':
    dataset = read_data(URL_DATA)
    print(dataset.shape)
    print(dataset[:5])
    dataset.to_csv('data\products_clean.csv',encoding='utf-8')