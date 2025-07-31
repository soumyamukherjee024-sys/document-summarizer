import streamlit as st
import matplotlib.pyplot as plt
from nlp_processor import (
    extract_text_from_pdf,
    generate_summary,
    generate_wordcloud,
    get_top_keywords,
    get_top_tfidf_words
)

st.set_page_config(layout="wide", page_title="Document Summarizer")

def main():
    st.title("📄 Document Summarization App")
    st.write("Upload a PDF document and get a summary, word cloud, and top keywords.")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        with st.spinner("Analyzing document... This may take a moment."):
            # Extract text from PDF
            raw_text = extract_text_from_pdf(uploaded_file)

            if raw_text and raw_text.strip():
                st.success("Document processed successfully!")

                # Generate and display results
                summary = generate_summary(raw_text)
                wordcloud = generate_wordcloud(raw_text)
                top_keywords = get_top_keywords(raw_text)
                top_tfidf_words = get_top_tfidf_words(raw_text)

                # Display Summary
                st.subheader("📝 Summary")
                st.write(summary)

                # Display Word Cloud and Keywords in columns
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("☁️ Word Cloud")
                    fig, ax = plt.subplots()
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis("off")
                    st.pyplot(fig)

                with col2:
                    st.subheader("🔑 Top 5 Keywords (Frequency)")
                    st.write(", ".join(top_keywords))

                    st.subheader("✨ Top 5 Relevant Words (TF-IDF)")
                    st.write(", ".join(top_tfidf_words))

                with st.expander("Show Full Extracted Text"):
                    st.text_area("Full Text", raw_text, height=300)
            else:
                st.error("Could not extract text from the PDF. The file might be empty, scanned, or corrupted.")

if __name__ == '__main__':
    main()

