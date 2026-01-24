import pandas as pd
import numpy as np
import re
import ssl
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer , WordNetLemmatizer
import logging

logger = logging.getLogger(__name__)
                       
def download_nltk_data():
    """
        Dowloading necessary nltk datasets
    """
    logger.info("Downloading the necessary nltk datasets")
    try:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('punkt_tab')
    except Exception as exception:
        logger.error(f"failed to download datasets from nltk:{exception}")
download_nltk_data()
logger.info("Required nltk datasets are downloaded sucessfully.")
class TextPreprocessing:
    def __init__(self, use_stemming = False,use_lemmatization = True):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.use_stemming = use_stemming
        self.use_lemmatization =use_lemmatization

    def do_lowercase(self,text):
        logger.info('Converting text into lowercase')
        return text.lower()
    
    def remove_html_tags(self,text):
        logger.info('removing html tags from the text')
        #  removing all html tags from the text[<br>,\,/,*,]
        clean_text = re.sub(r'<.*?>', '', text)
        return clean_text
    
    def remove_url(self,text):
        logger.info('removing url..')
        #removing links from the text
        clean_text = re.sub(r'http\S+|www\S+http\S+', '', text)
        return clean_text
    
    def remove_punctuation(self,text):
        logger.info('removing punctuation')
        cleaned_text = text.translate(str.maketrans('','',string.punctuation))
        return cleaned_text
    
    def remove_numbers(self,text):
        logger.info('removing numbers from the text')
        cleaned_text = re.sub(r'\d+', '',text)
        return cleaned_text
    
    def remove_spaces(self,text):
        logger.info('removing unwanted and extra spaces from the text')
        cleaned_text = ' '.join(text.split())
        return cleaned_text
    
    def tokenization(self,text):
        logger.info('converting text into tokens')
        tokens = word_tokenize(text)
        return tokens
    
    def remove_stopwords(self,tokens):
        logger.info('removing stopwords from tokens')
        filtered_tokens = [word for word in tokens if word not in self.stop_words]
        return filtered_tokens
    
    def do_stemming(self,tokens):
        logger.info('stemming the tokens')
        stemmed_tokens = [self.stemmer.stem(word) for word in tokens]
        return stemmed_tokens
    
    def do_lemmatization(self,tokens):
        logger.info('lemmatizing the tokens')
        lemmatized_tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        return lemmatized_tokens
    
    def preprocess_text(self,text):
        logger.info('Now performing the preprocess pipeline step by step')
        text = self.do_lowercase(text)
        text = self.remove_html_tags(text)
        text = self.remove_url(text)
        text = self.remove_punctuation(text)
        text = self.remove_numbers(text)
        text = self.remove_spaces(text)
        tokens = self.tokenization(text)
        tokens = self.remove_stopwords(tokens)
         
        if self.use_stemming:
            tokens = self.do_stemming(tokens)

        if self.use_lemmatization:
            tokens = self.do_lemmatization(tokens)

        clean_text = ' '.join(tokens)
        logger.info('Text preprocessing successfully done. Now you can see the clean text.')
        return clean_text
    
    def preprocess_dataframe(self, df, text_column='review'):
        
        df['cleaned_text'] = df[text_column].apply(
            lambda x: self.preprocess_text(x)
        )

        return df

    
if __name__ == "__main__":
    sample_text = """  THIS Movie review is about the film <b>Inception</b>, and I MUST say it is one of the BEST movies I have watched in 2024!!!.
    The story is running fast, runs through multiple dream levels, and keeps the viewer thinking, thinkers, and thinking again. 
    I watched it on 12/06/2024 and rated it 9/10, though some scenes felt too long   and confusing. 
    You can read more at https://www.imdb.com or check the trailer here <a href="https://youtube.com">Watch Now</a>. 
    The movie is, and always will be, a must-watch for sci-fi fans, but the sound effects #@$% and loud music at 120 dB were sometimes annoying. 
    Contact me at moviefan@email.com for more reviews.
                  """
                       
    preprocessor = TextPreprocessing(use_lemmatization=True)
    cleaned = preprocessor.preprocess_text(sample_text)
    print("\n --Preprocessed Text -- ")
    print(cleaned)

     






             

