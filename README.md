## Product Categorization
### Multi-Class Text Classification of products based on their description
 
### General info

The goal of the project is product categorization based on their description with Machine Learning and Deep Learning (MLP, CNN, Distilbert) algorithms. Additionaly I have created Doc2vec and Word2vec models, Topic Modeling (with LDA analysis) and EDA analysis (data exploration, data aggregation and cleaning data).

### Dataset
The dataset comes from http://makeup-api.herokuapp.com/ and has been obtained by an API. It can be seen at my previous project at [Extracting Data using API](https://github.com/aniass/Extracting-data-using-API).

The dataset contains the real data about makeup products such as brand, category, name and descriptions about makeup products where each description has been labeled with a specific product.

### Motivation

The aim of the project is multi-class text classification to make-up products based on their description. Based on given text as an input, one have predicted what would be the category. There are five types of categories corresponding to different makeup products. In analysis I used a different methods for a text representation (such as BoW +TF-IDF, doc2vec, Distilbert embeddings), feature extraction (Word2vec, Doc2vec) and various Machine Learning/Deep Lerning algorithms to get more accurate predictions and choose the most accurate one for our issue.  

### Project contains:
* Multi-class text classification with ML algorithms- ***Text_analysis.ipynb***
* Text classification with Distilbert model - ***Bert_products.ipynb***
* Text classification with MLP and Convolutional Neural Netwok (CNN) models - ***Text_nn.ipynb***
* Text classification with Doc2vec model -***Doc2vec.ipynb***
* Word2vec model - ***Word2vec.ipynb***
* LDA - Topic modeling - ***LDA_Topic_modeling.ipynb***
* EDA analysis - ***Products_analysis.ipynb***
* Python script to train ML models - **text_model.py**
* Python script to train ML models with smote method - **text_model_smote.py**
* Python script to text clean data - **clean_data.py**
* Python script to generate predictions from trained model - **predictions.py**
* data, models - data and models used in the project.

### Summary

To resolve problem of the product categorization based on their description I have applied multi-class text classification. I began with data analysis and data pre-processing from the dataset. Then I have used a combinations of text representation such as BoW +TF-IDF and doc2vec. I have experimented with several Machine Learning algorithms: Logistic Regression, Linear SVM, Multinomial Naive Bayes, Random Forest, Gradient Boosting and Neural Networks: MLP and Convolutional Neural Network (CNN) using different combinations of text representations and embeddings. Additionaly I have applied a transfer learning with  a pretrained Distilbert model from Huggingface Transformers library.

From the experiments one can see that the tested models give a overall high accuracy and similar results for the problem. The SVM (BOW +TF-IDF) model give the best accuracy of validation set equal to 96 %. Logistic regression performed very well both with BOW + TF-IDF and Doc2vec and achieved similar accuracy as MLP. CNN with word embeddings also has a very comparable result (93 %) to MLP. Transfer learning with Distilbert model also gave a similar results to previous models an we achieved an accuracy on the test set equal to 93 %. That shows the extensive models are not gave a better results to the problem than simple Machine Learning models such as SVM. 

Model | Embeddings | Accuracy
------------ | ------------- | ------------- 
SVC| BOW +TF-IDF  | 0.96
MLP| Word embedding  | 0.93
CNN | Word embedding | 0.93
Distilbert| Distilbert tokenizer | 0.93
Gradient Boosting | BOW +TF-IDF | 0.93
Random Forest| BOW +TF-IDF | 0.92
SVM | Doc2vec (DBOW)| 0.92
Logistic Regression | BOW +TF-IDF  | 0.91
Logistic Regression | Doc2vec (DM)  | 0.90
Naive Bayes | BOW +TF-IDF | 0.88


### Technologies
#### The project is created with:

* Python 3.6/3.8
* libraries: NLTK, gensim, Keras, TensorFlow, Hugging Face transformers, scikit-learn, pandas, numpy, seaborn, pyLDAvis.

#### Running the project:

To run this project use Jupyter Notebook or Google Colab.

You can run the scripts in the terminal:

    clean_data.py
    text_model.py
    text_model_smote.py

