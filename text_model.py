import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier


URL_DATA = 'data\\products_description.csv'

stop = stopwords.words('english')
porter = PorterStemmer()


def preprocess_data(text):
    ''' The function to remove punctuation,
    stopwords and apply stemming'''
    words = re.sub("[^a-zA-Z]", " ", text)
    words = [word.lower() for word in text.split() if word.lower() not in stop]
    words = [porter.stem(word) for word in words]
    return " ".join(words)


def read_data(path):
    ''' Function to read text data'''
    data = pd.read_csv(path, header=0, index_col=0)
    data['description'] = data['description'].apply(preprocess_data)
    X = data['description']
    y = data['product_type']
    return X, y


def prepare_data(X, y):
    ''' Function to split data on train and test set '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                        random_state=42)
    return X_train, X_test, y_train, y_test


def get_models(X_train, X_test, y_train, y_test):
    ''' Calculating models with score '''
    models = pd.DataFrame()
    classifiers = [
        LogisticRegression(),
        LinearSVC(),
        MultinomialNB(),
        RandomForestClassifier(n_estimators=50),
        GradientBoostingClassifier(n_estimators=50), ]

    for classifier in classifiers:
        pipeline = Pipeline(steps=[('vect', CountVectorizer(
                            min_df=5, ngram_range=(1, 2))),
                                   ('tfidf', TfidfTransformer()),
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


def main():
    X, y = read_data(URL_DATA)
    X_train, X_test, y_train, y_test = prepare_data(X, y)
    get_models(X_train, X_test, y_train, y_test)


if __name__ == '__main__':
    main()
