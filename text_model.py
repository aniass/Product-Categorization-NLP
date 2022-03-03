# Load libraries
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
import pickle


stop = stopwords.words('english')
porter = PorterStemmer()


def preprocess(text):
    ''' The function to remove punctuation,
    stopwords and apply stemming'''
    words = re.sub("[^a-zA-Z]", " ", text)
    words = [word.lower() for word in text.split() if word.lower() not in stop]
    words = [porter.stem(word) for word in words]
    return " ".join(words)


# Load dataset
url = 'data\\products_description.csv'
df = pd.read_csv(url, header=0, index_col=0)

# Shape
print(df.shape)
print(df.head())

df['description'] = df['description'].apply(preprocess)

# Separate into input and output columns
X = df['description']
y = df['product_type']

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                    random_state=42)

models = pd.DataFrame()

classifiers = [
    LogisticRegression(),
    LinearSVC(),
    MultinomialNB(),
    RandomForestClassifier(n_estimators=50),
    GradientBoostingClassifier(n_estimators=50),
    ]

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

# Save the model
with open('model.pkl', 'wb') as model_file:
    pickle.dump(pipeline, model_file)
