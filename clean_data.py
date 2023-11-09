# Load libraries
import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


URL_DATA = r'\data\products_description.csv'
CLEANED_DATA_PATH = r'data\products_clean.csv'


def grouping_data(df: pd.DataFrame) -> pd.DataFrame:
    """Group data to a smaller number of categories"""
    df.loc[df['product_type'].isin(['lipstick', 'lip_liner']), 'product_type'] = 'lipstick'
    df.loc[df['product_type'].isin(['blush', 'bronzer']), 'product_type'] = 'contour'
    df.loc[df['product_type'].isin(['eyeliner', 'eyeshadow', 'mascara', 'eyebrow']), 'product_type'] = 'eye_makeup'
    return df


def remove_punctuation(description: str) -> str:
    """Function to remove punctuation"""
    table = str.maketrans('', '', string.punctuation)
    return description.translate(table)


def remove_stopwords(text: str) -> str:
    """Function to remove stopwords"""
    stop = stopwords.words('english')
    text = [word.lower() for word in text.split() if word.lower() not in stop]
    return " ".join(text)


def stemmer(stem_text: str) -> str:
    """Function to apply stemming"""
    porter = PorterStemmer()
    stem_text = [porter.stem(word) for word in stem_text.split()]
    return " ".join(stem_text)


def preprocess_data(text_data: str) -> str:
    ''' Function to preprocess data'''
    data = grouping_data(text_data)
    data['description'] = data['description'].astype(str)
    data['description'] = data['description'].apply(remove_punctuation)
    data['description'] = data['description'].apply(remove_stopwords)
    data['description'] = data['description'].apply(stemmer)
    return data


def read_data(path: str) -> pd.DataFrame:
    """Function to read data"""
    try:
        df = pd.read_csv(path, header=0, index_col=0)
        return df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return pd.DataFrame()


if __name__ == '__main__':
    data = read_data(URL_DATA)
    data_clean = preprocess_data(data)
    if not data_clean.empty:
        print(data_clean.shape)
        print(data_clean.head(5))
        data_clean.to_csv(CLEANED_DATA_PATH, encoding='utf-8')