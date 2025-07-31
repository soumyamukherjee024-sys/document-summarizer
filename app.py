import streamlit as st
import os
from utils import (
    extract_text_from_pdf,
    extract_text_from_image,
    preprocess_text,
    generate_summary,
    get_top_keywords,
    get_top_tfidf_words,
    generate_wordcloud
)

def main():
    st.set_page_config(page_title="Document Summarizer", page_icon="📄", layout="wide")

    st.title("📄 Document Summarizer & Analyzer")
    st.markdown("""
        Upload a document (PDF or Image) to get a concise summary, a word cloud, 
        top keywords, and the most relevant words based on TF-IDF.
    """)

    # --- File Uploader ---
    uploaded_file = st.file_uploader(
        "Choose a PDF or Image file", 
        type=['pdf', 'png', 'jpg', 'jpeg']
    )

    if uploaded_file is not None:
        with st.spinner('Processing your document... Please wait.'):
            # --- Text Extraction ---
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            text = ""
            if file_extension == ".pdf":
                text = extract_text_from_pdf(uploaded_file)
            elif file_extension in [".png", ".jpg", ".jpeg"]:
                text = extract_text_from_image(uploaded_file)

        if not text.strip():
            st.error("Could not extract any text from the document. Please try another file.")
        else:
            st.success("Document processed successfully!")

            # --- Display Extracted Text ---
            with st.expander("View Extracted Text", expanded=False):
                st.text_area("Full Text", text, height=300)

            # --- NLP Analysis ---
            st.header("Analysis Results")
            
            # Preprocess text once for efficiency
            processed_tokens = preprocess_text(text)

            # --- Summary ---
            st.subheader("📜 Summary")
            summary = generate_summary(text)
            st.write(summary)

            # --- Keywords and Word Cloud in Columns ---
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("🔑 Top 5 Keywords")
                top_keywords = get_top_keywords(processed_tokens, top_n=5)
                st.table(top_keywords)

                st.subheader("✨ Top 5 TF-IDF Words")
                top_tfidf = get_top_tfidf_words(text, top_n=5)
                st.table(top_tfidf)

            with col2:
                st.subheader("☁️ Word Cloud")
                if processed_tokens:
                    wordcloud_image = generate_wordcloud(processed_tokens)
                    st.image(wordcloud_image, use_column_width=True)
                else:
                    st.warning("Not enough content to generate a word cloud.")

    else:
        st.info("Awaiting your document upload.")

if __name__ == '__main__':
    main()

