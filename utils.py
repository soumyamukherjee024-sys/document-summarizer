import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import spacy
from gensim.summarization import summarize as gensim_summarize
from wordcloud import WordCloud
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import nltk

# --- Configuration for Windows users ---
# If you're on Windows, you might need to specify the path to the Tesseract executable
# import os
# if os.name == 'nt':
#     pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# --- Model and Data Loading ---

@spacy.cache
def load_spacy_model(model_name="en_core_web_sm"):
    """Loads the spaCy model and caches it."""
    try:
        return spacy.load(model_name)
    except OSError:
        print(f"Downloading spaCy model '{model_name}'. This may take a moment.")
        spacy.cli.download(model_name)
        return spacy.load(model_name)

nlp = load_spacy_model()

try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("Downloading NLTK 'punkt' tokenizer.")
    nltk.download('punkt')

# --- Text Extraction Functions ---

def extract_text_from_pdf(file_stream):
    """Extracts text from a PDF file stream."""
    text = ""
    with fitz.open(stream=file_stream.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def extract_text_from_image(file_stream):
    """Extracts text from an image file stream using OCR."""
    image = Image.open(file_stream)
    text = pytesseract.image_to_string(image)
    return text

# --- NLP Processing Functions ---

def preprocess_text(text):
    """Cleans and preprocesses text using spaCy."""
    doc = nlp(text)
    # Lemmatize, remove stopwords and punctuation
    tokens = [
        token.lemma_.lower().strip() 
        for token in doc 
        if not token.is_stop and not token.is_punct and token.text.strip()
    ]
    return tokens

def generate_summary(text, ratio=0.2):
    """Generates a summary using Gensim's TextRank algorithm."""
    if not text or len(text.split()) < 50: # Gensim requires a minimum amount of text
        return "Text is too short to summarize."
    try:
        # Gensim's summarize function works on the raw text
        summary = gensim_summarize(text, ratio=ratio)
        return summary if summary else "Could not generate summary."
    except Exception as e:
        return f"An error occurred during summarization: {e}"

def get_top_keywords(tokens, top_n=5):
    """Gets the most frequent keywords from a list of tokens."""
    if not tokens:
        return []
    return [word for word, freq in Counter(tokens).most_common(top_n)]

def get_top_tfidf_words(text, top_n=5):
    """
    Gets the most relevant words using TF-IDF.
    We treat sentences as documents for single-document TF-IDF.
    """
    if not text:
        return []
    
    sentences = nltk.sent_tokenize(text)
    if len(sentences) < 2: # TF-IDF needs more than one document
        return get_top_keywords(preprocess_text(text), top_n)

    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(sentences)
    
    # Sum TF-IDF scores for each term across all sentences
    sum_tfidf = tfidf_matrix.sum(axis=0)
    tfidf_scores = pd.DataFrame(sum_tfidf, columns=vectorizer.get_feature_names_out()).T
    tfidf_scores.columns = ['tfidf']
    
    # Sort and get top N
    top_words = tfidf_scores.sort_values(by='tfidf', ascending=False).head(top_n)
    return top_words.index.tolist()


def generate_wordcloud(tokens):
    """Generates a word cloud image from a list of tokens."""
    if not tokens:
        return None
    
    text = " ".join(tokens)
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        colormap='viridis'
    ).generate(text)
    
    return wordcloud.to_image()

