import pandas as pd
from datasets import load_dataset
import os

def load_imdb_data(sample_size=None):
    """
    Load the dataset from the Hugging Face datasets library.
    """

    #Load the dataset
    dataset = load_dataset('imdb')

    #Convert the dataset to a pandas DataFrame
    train_df = pd.DataFrame(dataset['train'])
    test_df = pd.DataFrame(dataset['test'])

    #map the labels as 0=Negative and 1=Positive and created a column 'sentiment'
    train_df['sentiment'] = train_df['label'].map({0: 'negative', 1: 'positive'})
    test_df['sentiment'] = test_df['label'].map({0: 'negative', 1: 'positive'})

    #Renaming columns for better understanding
    train_df = train_df.rename(columns={'text': 'review'})
    test_df = test_df.rename(columns={'text': 'review'})

    #If samplesize is provided, take a sample of the specified size
    if sample_size:
        train_df = train_df.sample(n=min(sample_size, len(train_df)), random_state=42)
        test_df = test_df.sample(n=min(sample_size, len(test_df)), random_state=42)

        print(f"Loaded{len(train_df)} training sample.")
        print(f"Loaded {len(test_df)} testing sample.")

        return train_df, test_df

if __name__ == "__main__":
    train_df,test_df = load_imdb_data(sample_size=1000) 
    print(train_df.head())
    print(test_df.head())
