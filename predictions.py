from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from joblib import load


MODELS_PATH = '\models\SVC_model2.pkl'


def load_model():
    '''Load pretrained model'''
    try:
        with open(MODELS_PATH, 'rb') as file:
            model = load(file)
        return model
    except FileNotFoundError:
        print(f"Error: The model file '{MODELS_PATH}' was not found.")
        return None


def preprocess_data(text):
    '''Applying stopwords and stemming on raw data'''
    stop_words = set(stopwords.words('english'))
    porter = PorterStemmer()
    words = [porter.stem(word.lower()) for word in text if word.lower() not in stop_words]
    return words


def get_prediction(input_text):
    '''Generating predictions from raw data'''
    model = load_model()
    data = [input_text]
    processed_text =  preprocess_data(data)
    prediction = model.predict(processed_text)
    result = ''.join(prediction)
    print(f'Your product is in category: {result}')


if __name__ == '__main__':
    text = input("Type a your product description:\n")
    get_prediction(text)
    
