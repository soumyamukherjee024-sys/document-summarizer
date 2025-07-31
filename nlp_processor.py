import PyPDF2
import spacy
from collections import Counter
from string import punctuation
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

# Ensure necessary NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpus/stopwords')
except nltk.downloader.DownloadError:
    print("Downloading necessary NLTK data...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


def extract_text_from_pdf(pdf_file):
    """Extracts text from a given PDF file."""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        print(f"Error reading PDF file: {e}")
        return None

def generate_summary(text, top_n=5):
    """Generates an extractive summary of the text."""
    doc = nlp(text)
    stopwords_list = list(stopwords.words('english'))
    
    word_frequencies = {}
    for word in doc:
        if word.text.lower() not in stopwords_list and word.text.lower() not in punctuation:
            if word.text not in word_frequencies.keys():
                word_frequencies[word.text] = 1
            else:
                word_frequencies[word.text] += 1

    max_frequency = max(word_frequencies.values()) if word_frequencies else 0

    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word] / max_frequency)

    sentence_tokens = [sent for sent in doc.sents]
    sentence_scores = {}
    for sent in sentence_tokens:
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequencies[word.text.lower()]
                else:
                    sentence_scores[sent] += word_frequencies[word.text.lower()]

    from heapq import nlargest
    summarized_sentences = nlargest(top_n, sentence_scores, key=sentence_scores.get)
    final_sentences = [s.text for s in summarized_sentences]
    summary = " ".join(final_sentences)
    return summary

def generate_wordcloud(text):
    """Generates a word cloud from the text."""
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    return wordcloud

def get_top_keywords(text, top_n=5):
    """Extracts top N keywords based on frequency, ignoring stopwords."""
    doc = nlp(text)
    stopwords_list = list(stopwords.words('english'))
    
    # Lemmatize and filter tokens
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct and token.text.strip()]
    
    word_freq = Counter(tokens)
    top_keywords = [word for word, freq in word_freq.most_common(top_n)]
    return top_keywords

def get_top_tfidf_words(text, top_n=5):
    """Extracts top N most relevant words using TF-IDF."""
    # Use sentences as documents for TF-IDF
    sentences = sent_tokenize(text)
    if len(sentences) < 2: # TF-IDF requires more than one document
        return get_top_keywords(text, top_n) # Fallback to frequency-based keywords

    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(sentences)
    
    # Get feature names (words)
    feature_names = vectorizer.get_feature_names_out()
    
    # Sum TF-IDF scores for each term across all documents
    sum_tfidf = tfidf_matrix.sum(axis=0)
    tfidf_scores = [(feature_names[i], sum_tfidf[0, i]) for i in range(len(feature_names))]
    
    # Sort by score and get top N
    sorted_tfidf_scores = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)
    top_words = [word for word, score in sorted_tfidf_scores[:top_n]]
    
    return top_words

