import nltk

nltk.download(['punkt','stopwords','wordnet'])
nltk.download('averaged_perceptron_tagger')

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import sys
import os
import re
import pickle
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer 
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection  import GridSearchCV
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.metrics import precision_recall_fscore_support,accuracy_score,label_ranking_average_precision_score



def load_data(db_path='../data/disaster.db',tablename='_table'):
    """
    Load data from database and return X and y.
    Args:
      db_path(str) -> database file name included path
      tablename:(str) -> table name in the database file.
    Return:
      X(pd.DataFrame) -> messages for X
      y(pd.DataFrame) -> labels part in messages for y
    """
    # load data from database
    engine = create_engine('sqlite:///'+db_path)
    table_name = os.path.basename(db_path).replace(".db","") + "_table"
    df = pd.read_sql_table(table_name,engine)
    
    ## clean data
    df = df.drop(['child_alone'],axis=1)
    # define features and label arrays
    df['related']=df['related'].map(lambda x: 1 if x == 2 else x)
    X = df['message']
    y = df.iloc[:,4:]
    return X, y


def tokenize(text,url_place_holder_string="urlplaceholder"):
    """
    Tokenize the text function
    
    Args:
        text -> Text message which needs to be tokenized
    Return:
        clean_tokens -> List of tokens extracted from the provided text
    """
    
    # Replace all urls with a urlplaceholder string
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # Extract all the urls from the provided text 
    detected_urls = re.findall(url_regex, text)
    
    # Replace url with a url placeholder string
    for detected_url in detected_urls:
        text = text.replace(detected_url, url_place_holder_string)

    # Extract the word tokens from the provided text
    tokens = nltk.word_tokenize(text)
    
    #Lemmanitizer to remove inflectional and derivationally related forms of a word
    lemmatizer = nltk.WordNetLemmatizer()

    # List of clean tokens
    clean_tokens = [lemmatizer.lemmatize(w).lower().strip() for w in tokens]
    return clean_tokens


class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Starting Verb Extractor class
    
    This class extract the starting verb of a sentence,
    creating a new feature for the ML classifier
    """

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    # Given it is a tranformer we can return the self 
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def build_model():
    """
    Build model.
    Return:
        pipline: sklearn.model_selection.GridSearchCV. It contains a sklearn estimator.
    """
    # text processing and model pipeline
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('count_vectorizer', CountVectorizer(tokenizer=tokenize)),
                ('tfidf_transformer', TfidfTransformer())
            ])),

            ('starting_verb_transformer', StartingVerbExtractor())
        ])),

        ('classifier', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    # define parameters for GridSearchCV
    parameters = {'classifier__estimator__learning_rate': [0.01, 0.02, 0.05],
              'classifier__estimator__n_estimators': [10, 20, 40]}

    # create gridsearch object and return as final model pipeline
    cv = GridSearchCV(estimator=pipeline, param_grid=parameters, cv=3, scoring='f1_weighted', verbose=3)

    return cv

def train(X, y, model):
    # train test split
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=42)

    # fit model
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test) 
    
    # output model test results
    evaluate_model(y_test, y_pred)

    return model


def evaluate_model(y_test, y_pred):
    result=precision_recall_fscore_support(y_test, y_pred)
    for i, col in enumerate(y_test.columns.values): #category_names
        accu=accuracy_score(y_test.loc[:,col],y_pred[:,i])
        print('{}\n Accuracy: {:.4f}    % Precision: {:.4f}   % Recall {:.4f} '.format(
            col,accu,result[0][i],result[1][i]))
    avg_precision = label_ranking_average_precision_score(y_test, y_pred)
    avg_score= ('label ranking average precision: {}'.format(avg_precision))
    print(avg_score)


def export_model(cv):
    """
    save model as pickle file.
    Args:
      cv:target model
    Return:
      N/A
    """
    
    with open('classifier.pkl', 'wb') as file:
        pickle.dump(cv, file)
        

def run_pipeline(data_file):
    print("Load data")
    X, y = load_data(data_file)  # run ETL pipeline
    print("Build model")
    model = build_model()  # build model pipeline
    print("Train model")
    model = train(X, y, model)  # train model pipeline
    print("Export model")
    export_model(model)  # save model

def main():
    if len(sys.argv) == 2:
        data_file = sys.argv[1]  # get filename of dataset
        run_pipeline(data_file)  # run data pipeline

        print('Trained model saved!')
    else: 
        print("Please provide the arguments correctly: \n\
            1) Path to SQLite database \n\
            Ex:python train_classifier.py ../data/disaster.db ")


if __name__ == '__main__':
    main()
    