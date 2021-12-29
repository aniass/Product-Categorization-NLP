# Load libraries
import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

stop = stopwords.words('english')
porter = PorterStemmer()


def remove_punctuation(description):
    """The function to remove punctuation"""
    table = str.maketrans('', '', string.punctuation)
    return description.translate(table)


def remove_stopwords(text):
    """The function to removing stopwords"""
    text = [word.lower() for word in text.split() if word.lower() not in stop]
    return " ".join(text)


def stemmer(stem_text):
    """The function to apply stemming"""
    stem_text = [porter.stem(word) for word in stem_text.split()]
    return " ".join(stem_text)


# Load dataset
url = 'C:\\Python Scripts\\API_products\\products_description.csv'
dataset = pd.read_csv(url, header=0, index_col=0)

# shape
print(dataset.shape)
print(dataset.head(5))

dataset['description'] = dataset['description'].astype(str)
dataset['description'] = dataset['description'].apply(remove_punctuation)
dataset['description'] = dataset['description'].apply(remove_stopwords)
dataset['description'] = dataset['description'].apply(stemmer)
print(dataset[:5])

data = dataset.to_csv('C:\\Python Scripts\\API_products\\products_clean.csv',
                      encoding='utf-8')
