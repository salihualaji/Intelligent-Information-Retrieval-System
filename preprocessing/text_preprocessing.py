import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    
    tokens = word_tokenize(text.lower())
    
    filtered_tokens = [
        word for word in tokens if word.isalnum() and word not in stop_words
    ]
    
    return " ".join(filtered_tokens)