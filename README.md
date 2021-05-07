## Multi-Class Text Classification of products based on their description
 
### General info

The goal of the project is product categorization based on their description with Machine Learning and Deep Learning (MLP, CNN) algorithms. Additionaly we have created Doc2vec and Word2vec models, Topic Modeling (with LDA analysis) and EDA analysis (data exploration, data aggregation and cleaning data).

The dataset comes from http://makeup-api.herokuapp.com/ and has been obtained from my previous project at [Extracting Data using API](https://github.com/aniass/Extracting-data-using-API).

### Motivation

The aim of the project is multi-class text classification to make-up products based on their description. Based on given text as an input, we have predicted what would be the category. We have five types of categories corresponding to different makeup products. In our analysis we used a different  methods for a feature extraction (such as Word2vec, Doc2vec) and various Machine Learning/Deep Lerning algorithms to get more accurate predictions and choose the most accurate one for our issue. 

### Project contains:
* Multi-class text classification with ML algorithms- ***Text_analysis.ipynb***
* Text classification with Doc2vec model -***Doc2vec.ipynb***
* Text classification with MLP and Convolutional Neural Netwok (CNN) models - ***Text_nn.ipynb***
* EDA analysis - ***Products_analysis.ipynb***
* LDA - Topic modeling - ***LDA_Topic_modeling.ipynb***
* Word2vec model - ***Word2vec.ipynb***
* Python scripts to clean data and use text model - **clean_data.py, text_model.py**

### Summary

We begin with data analysis and data pre-processing from our dataset. Then we have used a few combination of text representation such as BoW and TF-IDF and we have trained the word2vec and doc2vec models from our data. We have experimented with several Machine Learning algorithms: Logistic Regression, Linear SVM, Multinomial Naive Bayes, Random Forest, Gradient Boosting and MLP and Convolutional Neural Network (CNN) using different combinations of text representations and embeddings. 

From our experiments we can see that the tested models give a overall high accuracy and similar results for our  problem. The SVM (BOW +TF-IDF) model and MLP model give the best  accuracy of validation set. Logistic regression performed very well both with BOW +TF-IDF and Doc2vec and achieved similar accuracy as MLP. CNN with word embeddings also has a very comparable result (0.93) to MLP. 


Model | Embeddings | Accuracy
------------ | ------------- | ------------- 
Logistic Regression | BOW +TF-IDF | 0.91
SVM | BOW +TF-IDF | 0.93
Naive Bayes | BOW +TF-IDF | 0.90
Random Forest | BOW +TF-IDF | 0.91
Gradient Boosting | BOW +TF-IDF | 0.91
Logistic Regression | Doc2vec (DBOW) | 0.91
Logistic Regression | Doc2vec (DM) | 0.89
SVM | Doc2vec (DBOW) | 0.93
MLP| Word embedding | 0.93
CNN | Word embedding | 0.93


#### The project is created with:

* Python 3.6
* libraries: NLTK, Gensim, Keras, Scikit-learn, Pandas, Numpy, Seaborn, pyLDAvis.

#### Running the project:

* To run this project use Jupyter Notebook or Google Colab.
