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


def grouping_data(df: pd.DataFrame) -> pd.DataFrame:
    """Grouping data to a smaller number of categories"""
    df.loc[df['product_type'].isin(['lipstick','lip_liner']),'product_type'] = 'lipstick'
    df.loc[df['product_type'].isin(['blush','bronzer']),'product_type'] = 'contour'
    df.loc[df['product_type'].isin(['eyeliner','eyeshadow','mascara','eyebrow']),'product_type'] = 'eye_makeup'
    return df


def preprocess_data(text: str) -> str:
    """The function to remove punctuation, stopwords and apply stemming"""
    stop = stopwords.words('english')
    porter = PorterStemmer()
    words = re.sub("[^a-zA-Z]", " ", text)
    words = [word.lower() for word in text.split() if word.lower() not in stop]
    words = [porter.stem(word) for word in words]
    return " ".join(words)


def read_data(path: str) -> pd.DataFrame:
    """Function to read text data"""
    df = pd.read_csv(path, header=0, index_col=0)
    data = grouping_data(df)
    data['description'] = data['description'].apply(preprocess_data)
    return data


def splitting_data(data: pd.DataFrame):
    """Function to split data on train and test set"""
    X = data['description']
    y = data['product_type']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                        random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test


def get_models(X_train, X_test, y_train, y_test)-> pd.DataFrame:
    """Calculating models with score"""
    models = pd.DataFrame()
    classifiers = [
        LogisticRegression(),
        LinearSVC(),
        MultinomialNB(),
        RandomForestClassifier(n_estimators=50),
        GradientBoostingClassifier(n_estimators=50), ]

    for classifier in classifiers:
        try:
            pipeline = Pipeline(steps=[
                    ('vect', CountVectorizer(min_df=5, ngram_range=(1, 2))),
                    ('tfidf', TfidfTransformer()),
                    ('classifier', classifier)
            ])
            pipeline.fit(X_train, y_train)
            score = pipeline.score(X_test, y_test)
            param_dict = {
                      'Model': classifier.__class__.__name__,
                      'Score': score
            }
            models = models.append(pd.DataFrame(param_dict, index=[0]))
        except Exception as e:
            print(f"Error occurred while fitting {classifier.__class__.__name__}: {str(e)}")

    models.reset_index(drop=True, inplace=True)
    models_sorted = models.sort_values(by='Score', ascending=False)
    print(models_sorted)
    return models_sorted


if __name__ == '__main__':
    df = read_data(URL_DATA)
    X_train, X_test, y_train, y_test = splitting_data(df)
    get_models(X_train, X_test, y_train, y_test)
