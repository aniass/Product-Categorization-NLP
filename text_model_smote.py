'''Product categorization: the second approach to text
   classification with SMOTE method '''

import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imbpipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


URL_DATA = 'data\\products_description.csv'

stop = stopwords.words('english')
porter = PorterStemmer()


def grouping_data(df):
    """Grouping data to a smaller number of categories"""
    df.loc[df['product_type'].isin(['lipstick','lip_liner']),'product_type'] = 'lipstick'
    df.loc[df['product_type'].isin(['blush','bronzer']),'product_type'] = 'contour'
    df.loc[df['product_type'].isin(['eyeliner','eyeshadow','mascara','eyebrow']),'product_type'] = 'eye_makeup'
    return df


def preprocess_data(text):
    ''' The function to remove punctuation,
    stopwords and apply stemming'''
    words = re.sub("[^a-zA-Z]", " ", text)
    words = [word.lower() for word in text.split() if word.lower() not in stop]
    words = [porter.stem(word) for word in words]
    return " ".join(words)


def read_data(path):
    ''' Function to read text data'''
    df = pd.read_csv(path, header=0, index_col=0)
    data = grouping_data(df)
    data['description'] = data['description'].apply(preprocess_data)
    return data


def splitting_data(data):
    '''Function to split data on train and test set'''
    X = data['description']
    y = data['product_type']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                        random_state=42)
    return X_train, X_test, y_train, y_test


def get_models(X_train, X_test, y_train, y_test):
    '''Calculating models with score'''
    models = pd.DataFrame()
    classifiers = [
        LogisticRegression(),
        LinearSVC(),
        MultinomialNB(),
        RandomForestClassifier(n_estimators=50),
        GradientBoostingClassifier(n_estimators=50), ]

    for classifier in classifiers:
        pipeline = imbpipeline(steps=[('vect', CountVectorizer(
                               min_df=5, ngram_range=(1, 2))),
                                      ('tfidf', TfidfTransformer()),
                                      ('smote', SMOTE()),
                                      ('classifier', classifier)])
        pipeline.fit(X_train, y_train)
        score = pipeline.score(X_test, y_test)
        param_dict = {
                      'Model': classifier.__class__.__name__,
                      'Score': score
                     }
        models = models.append(pd.DataFrame(param_dict, index=[0]))

    models.reset_index(drop=True, inplace=True)
    print(models.sort_values(by='Score', ascending=False))


if __name__ == '__main__':
    df = read_data(URL_DATA)
    X_train, X_test, y_train, y_test = splitting_data(df)
    get_models(X_train, X_test, y_train, y_test)
