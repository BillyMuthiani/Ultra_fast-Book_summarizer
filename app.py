import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.neighbors import NearestNeighbors
from sklearn.base import BaseEstimator, TransformerMixin
from difflib import get_close_matches
from rapidfuzz import fuzz
import re
import fitz  # PyMuPDF
import nltk
import unicodedata
from fpdf import FPDF
from nltk.tokenize import sent_tokenize
from time import time
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import streamlit as st
from gtts import gTTS
from piper import PiperVoice
#import piper
import cohere
import warnings


nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
warnings.filterwarnings("ignore", category=DeprecationWarning)

DATASET_PATH = './dataset/Goodreads_BestBooksEver_1-10000.csv'
PDF_SUMMARY_PATH = './summarized_pdf/'
MP3_SUMMARY_PATH = './summarized_mp3/'
TTS_MODEL_PATH = './tts_model/en_US-bryce-medium.onnx'


# Placeholder pre-summarized text (replace with your Cohere-generated summary)
PRE_SUMMARIZED_TEXT = """
The Alchemist by Paulo Coelho follows Santiago, a young Andalusian shepherd, who dreams of a treasure hidden near the pyramids in Egypt. Guided by omens, he embarks on a journey of self-discovery, leaving his comfortable life to pursue his Personal Legend. Along the way, he meets a gypsy, a king, and an alchemist, who teach him to follow his heart and embrace the unknown. Santiago faces challenges, including theft and danger in the desert, but learns that true wealth lies in the journey and understanding the interconnectedness of all things. The story explores themes of destiny, courage, and listening to one’s dreams. In the end, Santiago discovers the treasure was buried back home, but the journey transforms him, revealing the importance of pursuing one’s purpose.
"""

# Load Cohere API Key from Streamlit secrets
COHERE_API_KEY = st.secrets.get("COHERE_API_KEY", os.getenv("COHERE_API_KEY"))
if not COHERE_API_KEY:
    st.error("Cohere API key not found. Please set it in Streamlit secrets or environment variables.")
    st.stop()
co = cohere.Client(COHERE_API_KEY)

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv(DATASET_PATH)
    df.drop_duplicates(inplace=True)
    return df

df = load_data()

# Genre Cleaning Function
def clean_genres(genre_str):
    if pd.isna(genre_str):
        return ""
    genre_str = re.sub(r"[\[\]{}()]", "", genre_str)
    genre_list = re.split(r"[,|/]", genre_str)
    genre_list = [g.strip().split("-")[0].split("/")[0] for g in genre_list if g.strip()]
    genre_list = list(dict.fromkeys(genre_list))
    return "|".join(genre_list[:3])

# Custom Transformers
class ColumnExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.column]

# Book Recommender Class
class BookRecommender:
    def __init__(self, n_neighbors=10, fuzzy_threshold=80):
        self.n_neighbors = n_neighbors
        self.fuzzy_threshold = fuzzy_threshold
        self.model = NearestNeighbors(metric='cosine')
        self.feature_pipeline = None
        self.books_df = None
        self.indices = None

    def fit(self, df):
        df = df.copy()
        df['cleanGenres'] = df['bookGenres'].apply(clean_genres)
        df = df.dropna(subset=['bookTitle', 'bookAuthors', 'bookRating'])
        self.books_df = df.reset_index(drop=True)
        self.indices = pd.Series(self.books_df.index, index=self.books_df['bookTitle']).drop_duplicates()

        title_pipe = Pipeline([('extract', ColumnExtractor('bookTitle')), ('tfidf', TfidfVectorizer())])
        author_pipe = Pipeline([('extract', ColumnExtractor('bookAuthors')), ('tfidf', TfidfVectorizer())])
        genre_pipe = Pipeline([('extract', ColumnExtractor('cleanGenres')), ('tfidf', TfidfVectorizer())])
        numeric_pipe = Pipeline([('extract', ColumnExtractor(['bookRating', 'ratingCount', 'reviewCount'])), ('scale', MinMaxScaler())])

        self.feature_pipeline = FeatureUnion([('title', title_pipe), ('author', author_pipe), ('genres', genre_pipe), ('numeric', numeric_pipe)])
        features = self.feature_pipeline.fit_transform(self.books_df)
        self.model.fit(features)
        return self

    def _find_closest_title(self, title):
        all_titles = self.books_df['bookTitle'].tolist()
        matches = get_close_matches(title, all_titles, n=1, cutoff=0)
        if matches:
            score = fuzz.ratio(matches[0].lower(), title.lower())
            return matches[0], score
        return None, 0

    def recommend(self, title, n_recommendations=5, auto_accept_closest=True):
        if title not in self.indices:
            closest, score = self._find_closest_title(title)
            if score >= self.fuzzy_threshold and auto_accept_closest:
                st.info(f"Using closest match: '{closest}' (score: {score})")
                title = closest
            else:
                raise ValueError(f"Book title '{title}' not found. Closest match '{closest}' (score: {score}).")

        idx = self.indices[title]
        if isinstance(idx, pd.Series):
            idx = idx.iloc[0]
        vec = self.feature_pipeline.transform(self.books_df.iloc[idx:idx + 1])

        distances, indices = self.model.kneighbors(vec, n_neighbors=self.n_neighbors + 10)
        rec_idxs = indices.flatten()

        input_title_clean = title.lower().strip()
        seen_titles = set()
        recommendations = []

        for i in rec_idxs:
            candidate_title = self.books_df.iloc[i]['bookTitle']
            if fuzz.ratio(candidate_title.lower().strip(), input_title_clean) < 90:
                if candidate_title not in seen_titles:
                    seen_titles.add(candidate_title)
                    recommendations.append(i)
            if len(recommendations) >= n_recommendations:
                break

        return self.books_df.iloc[recommendations][['bookTitle', 'bookAuthors', 'bookRating', 'ratingCount', 'reviewCount', 'cleanGenres']].reset_index(drop=True)

# Text Extraction Function
def extract_text(file):
    try:
        if file.name.lower().endswith(".txt"):
            return unicodedata.normalize("NFKC", file.read().decode("utf-8").replace("\ufffd", ""))
        elif file.name.lower().endswith(".pdf"):
            doc = fitz.open(stream=file.read(), filetype="pdf")
            text = "".join(page.get_text() for page in doc)
            doc.close()
            return unicodedata.normalize("NFKC", text.replace("\ufffd", ""))
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return ""

# Chunking Function
def chunk_sentences(sentences, max_words=1500):
    chunks = []
    current = []
    total_words = 0
    for sentence in sentences:
        word_count = len(sentence.split())
        if total_words + word_count > max_words:
            chunks.append(" ".join(current))
            current = [sentence]
            total_words = word_count
        else:
            current.append(sentence)
            total_words += word_count
    if current:
        chunks.append(" ".join(current))
    return chunks

# Text-to-Speech Function
def generate_tts(text, output_file):
    try:
        if not text.strip():
            st.warning("No text provided for TTS generation.")
            return False
        tts = gTTS(text=text, lang='en')
        tts.save(output_file)
        print(f"Saved audio to {output_file}")
        return True
    except Exception as e:
        print(f"Error generating audio: {e}")
        return False


# Streamlit App
def main():
    st.title("Book Recommender and Text Summarizer")
    st.markdown("Enter a book title for recommendations or upload a PDF/TXT file for summarization. Use pre-summarized text for demo reliability.")

    # Initialize session state
    if 'recommender' not in st.session_state:
        with st.spinner("Initializing book recommender..."):
            st.session_state.recommender = BookRecommender(n_neighbors=10).fit(df)

    # Book Recommendation Section
    st.header("Book Recommendations")
    query = st.text_input("Enter a book title:", placeholder="e.g., The Alchemist")
    if st.button("Get Recommendations") and query:
        try:
            results = st.session_state.recommender.recommend(query, n_recommendations=5)
            st.subheader(f"Top recommendations for '{query}':")
            results_display = results[['bookTitle', 'bookAuthors', 'bookRating', 'cleanGenres']].copy()
            results_display['bookRating'] = results_display['bookRating'].round(2)
            results_display.reset_index(inplace=True)
            results_display.index += 1
            st.dataframe(results_display, use_container_width=True)
        except ValueError as e:
            st.error(str(e))

    # File Upload and Summarization Section
    st.header("Text Summarization")
    use_pre_summarized = st.radio("Use pre-summarized text for demo?", ("Yes", "No"), index=0)
    uploaded_file = None
    summary_text = PRE_SUMMARIZED_TEXT if use_pre_summarized == "Yes" else None
    input_file = "pre_summarized_text.txt" if use_pre_summarized == "Yes" else None

    if use_pre_summarized == "No":
        uploaded_file = st.file_uploader("Upload a PDF or TXT file to summarize:", type=["pdf", "txt"])
        if uploaded_file:
            input_file = uploaded_file.name
            st.write(f"Processing file: {input_file}")
            text = extract_text(uploaded_file)
            if not text:
                st.error("No text extracted from the file.")
                return

            sentences = sent_tokenize(text)
            chunks = chunk_sentences(sentences)
            st.write(f"Total Chunks: {len(chunks)}")

            summaries = []
            progress_bar = st.progress(0)
            start_time = time()
            for i, chunk in enumerate(chunks):
                try:
                    response = co.chat(
                        model="command-r-plus",
                        message=f"Summarize this text in simple, concise language:\n\n{chunk[:3000]}",
                        temperature=0.3,
                    )
                    summaries.append(response.text.strip())
                except Exception as e:
                    st.warning(f"Error at chunk {i}: {e}")
                    summaries.append("[Summary unavailable]")
                progress_bar.progress((i + 1) / len(chunks))

            summary_text = "\n\n".join(summaries)
            elapsed = time() - start_time
            st.success(f"Done summarizing in {elapsed:.2f} seconds")

    if summary_text:
        # Display summary
        st.subheader("Summary")
        st.text_area("Summary Text", summary_text, height=200)

        # Truncate summary for PDF
        if len(summary_text) > 7000:
            summary_text = summary_text[:7000].rsplit(" ", 1)[0] + "..."
            summary_text += "\n\n[Note: Summary truncated to fit 2 pages.]"

        # Generate PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, f"Summary of {input_file}", ln=True, align="C")
        pdf.set_font("Arial", size=10)
        summary_text = unicodedata.normalize("NFKC", summary_text)
        summary_text = summary_text.replace("\u2014", "-").replace("\u2019", "'")
        pdf.ln(10)
        pdf.multi_cell(0, 6, summary_text)

        output_path = f'{PDF_SUMMARY_PATH}{input_file.replace(".pdf", "_summary.pdf").replace(".txt", "_summary.pdf")}'
        pdf.output(output_path)

        # Text-to-Speech Option
        st.header("Audio Generation")
        tts_choice = st.radio("Generate audio version of the summary?", ("Yes", "No"))
        audio_path = None
        if tts_choice == "Yes":
            audio_path = f'{MP3_SUMMARY_PATH}{input_file.replace(".pdf", "_summary.mp3").replace(".txt", "_summary.mp3")}'
            print(f"Generating audio file at: {audio_path}")
            print(f"Summary text length: {summary_text} characters")
            if generate_tts(summary_text, audio_path):
                with open(audio_path, "rb") as f:
                    st.audio(f, format="audio/mp3")

        # Email Sending Section
        st.header("Send Summary via Email")
        sender_email = st.secrets.get("SENDER_EMAIL", "your_email@gmail.com")
        password = st.secrets.get("SENDER_PASSWORD", "your_app_password")
        receiver_email = st.text_input("Enter receiver's email address:", placeholder="example@domain.com")
        if not receiver_email:
            st.warning("Please enter a receiver's email address to send the summary.")
        if not sender_email or not password:
            st.error("Sender email and password are required to send the summary.")
            return    
           
        if st.button("Send Email") and receiver_email:
            try:
                msg = MIMEMultipart()
                msg['From'] = sender_email
                msg['To'] = receiver_email
                msg['Subject'] = f"Summary of {input_file}"

                body = f"Hi,\n\nPlease find the attached PDF with the summary of '{input_file}'."
                if audio_path and os.path.exists(audio_path):
                    body += " An audio version of the summary is also attached."
                body += "\n\nBest regards,\n"
                msg.attach(MIMEText(body, 'plain'))

                # Attach PDF
                with open(output_path, "rb") as attachment:
                    part = MIMEBase("application", "octet-stream")
                    part.set_payload(attachment.read())
                    encoders.encode_base64(part)
                    part.add_header(
                        "Content-Disposition",
                        f"attachment; filename= {os.path.basename(output_path)}",
                    )
                    msg.attach(part)

                # Attach audio if generated
                if audio_path and os.path.exists(audio_path):
                    with open(audio_path, "rb") as attachment:
                        part = MIMEBase("application", "octet-stream")
                        part.set_payload(attachment.read())
                        encoders.encode_base64(part)
                        part.add_header(
                            "Content-Disposition",
                            f"attachment; filename= {os.path.basename(audio_path)}",
                        )
                        msg.attach(part)

                server = smtplib.SMTP('smtp.gmail.com', 587) 
                server.starttls()
                server.login(sender_email, password)
                server.send_message(msg)
                st.success("Email sent with PDF attachment" + (" and audio attachment!" if audio_path else "!"))
                server.quit()
            except Exception as e:
                st.error(f"Error sending email: {e}")
                st.warning("Email failed, but summary and audio (if generated) are available above.")

if __name__ == "__main__":
    main()