from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from joblib import load


MODELSPATH = '\models\SVC_model2.pkl'

stop = stopwords.words('english')
porter = PorterStemmer()


def load_model():
    '''Loading pretrained model'''
    with open(MODELSPATH, 'rb') as file:
        model = load(file)
        return model


def preprocess_data(text):
    ''' Applying stopwords and stemming on raw data'''
    words = [word.lower() for word in text if word.lower() not in stop]
    words = [porter.stem(word) for word in words]
    return words


def get_prediction(input_text):
    ''' Generating predictions from raw data'''
    model = load_model()
    data = [input_text]
    text =  preprocess_data(data)
    prediction = model.predict(text)
    result = ''.join(prediction)
    print('---------------')
    print(f'Your product is in category: {result}')


if __name__ == '__main__':
    text = input("Type a your product description:\n")
    get_prediction(text)
    