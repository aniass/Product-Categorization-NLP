# Load libraries
import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

stop = stopwords.words('english')
porter = PorterStemmer()

URL_DATA = '\data\products_description.csv'


def grouping_data(df):
    """Grouping data to a smaller number of categories"""
    df.loc[df['product_type'].isin(['lipstick','lip_liner']),'product_type'] = 'lipstick'
    df.loc[df['product_type'].isin(['blush','bronzer']),'product_type'] = 'contour'
    df.loc[df['product_type'].isin(['eyeliner','eyeshadow','mascara','eyebrow']),'product_type'] = 'eye_makeup'
    return df


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
    df = pd.read_csv(path, header=0, index_col=0)
    data = grouping_data(df)
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