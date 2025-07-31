# 📄 Document Summarization App

This is a Python application with a Streamlit UI that summarizes PDF documents using NLP.

## Features

- Upload a PDF document.
- Get an extractive summary of the document.
- View a word cloud of the most frequent words.
- See the top 5 keywords based on frequency.
- See the top 5 most relevant words based on TF-IDF scores.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd doc-summarizer
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```
    *The first time you run the app, it will automatically download the necessary `spaCy` and `NLTK` models if they are not found.*

## How to Run the App

1.  **Navigate to the project directory and run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

2.  **Open your web browser and go to the local URL provided by Streamlit (usually `http://localhost:8501`).**

