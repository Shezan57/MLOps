import numpy as np
import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer


# fetch the data from data/raw
train_data = pd.read_csv("./data/raw/train.csv")
test_data = pd.read_csv("./data/raw/test.csv")

# transform the data
nltk.download('stopwords')
nltk.download('wordnet')

def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    text = text.split()
    text = [word for word in str(text).split() if word not in stop_words]
    return " ".join(text)

def removing_numbers(text):
    text = ''.join([i for i in text if not i.isdigit()])
    return text

def lowercase(text):
    text = text.split()
    text = [word.lower() for word in text]
    return " ".join(text)

def removing_punctuations(text):
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
    text = text.replace('؛',"", )
    return text.strip()

def removing_url(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_small_sentences(df):
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan

def normalize_text(df):
    df.content = df.content.apply(lambda content: lowercase(content))
    df.content = df.content.apply(lambda content: remove_stopwords(content))
    df.content = df.content.apply(lambda content: removing_numbers(content))
    df.content = df.content.apply(lambda content: removing_punctuations(content))
    df.content = df.content.apply(lambda content: removing_url(content))
    df.content = df.content.apply(lambda content: lemmatization(content))
    return df

# def normalized_sentence(sentence):
#     sentence= lowercase(sentence)
#     sentence= remove_stopwords(sentence)
#     sentence= removing_numbers(sentence)
#     sentence= removing_punctuations(sentence)
#     sentence= removing_url(sentence)
#     sentence= lemmatization(sentence)
#     return sentence

train_processed_data = normalize_text(train_data)
test_processed_data = normalize_text(test_data)

# store the data inside data/processed
data_path = os.path.join("./data", "processed")
os.makedirs(data_path, exist_ok=True)
train_processed_data.to_csv(os.path.join(data_path, "train_processed_data.csv"), index=False)
test_processed_data.to_csv(os.path.join(data_path, "test_processed_data.csv"), index=False)