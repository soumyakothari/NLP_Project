import pandas as pd
import re
import string
import unicodedata
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import spacy
import streamlit as st
from spacy.cli import download

download("en_core_web_sm")

class BookRecommender:
    def __init__(self, csv_path: str):
        # Load data
        self.df = pd.read_csv(csv_path)
        # NLP resources
        self.nlp = spacy.load('en_core_web_sm')
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = TfidfVectorizer(stop_words='english')
        # Prepare dataset
        self._prepare()

    def preprocess_text(self, text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            return ""
        # Normalize and lowercase
        text = unicodedata.normalize('NFKD', text)
        text = text.lower()
        # Remove non-letter characters
        text = re.sub(r"[^a-z]", " ", text)
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Tokenize
        tokens = WordPunctTokenizer().tokenize(text)
        # Remove stop words
        tokens = [t for t in tokens if t not in self.stop_words]
        # Lemmatize
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        # Filter short tokens
        tokens = [t for t in tokens if len(t) > 2]
        return ' '.join(tokens)

    def _prepare(self):
        # Combine title and genres
        self.df['content'] = self.df['Title'].fillna('') + ' ' + self.df['genres'].fillna('')
        # Clean content
        self.df['Cleaned_Content'] = self.df['content'].apply(self.preprocess_text)
        # Vectorize
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['Cleaned_Content'])
        # Compute similarity matrix
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        # Build reverse lookup
        self.indices = pd.Series(self.df.index, index=self.df['Title'].str.lower()).drop_duplicates()

    def recommend(self, title: str, num_recommendations: int = 5):
        key = title.lower()
        if key not in self.indices:
            return []
        idx = self.indices[key]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]
        book_indices = [i for i, _ in sim_scores]
        return self.df['Title'].iloc[book_indices].tolist()

    def analyze_sentiment(self, title: str):
        key = title.lower()
        if key not in self.indices:
            return None
        idx = self.indices[key]
        text = self.df.loc[idx, 'Cleaned_Content']
        polarity = TextBlob(text).sentiment.polarity
        if polarity > 0:
            return 'Positive'
        elif polarity < 0:
            return 'Negative'
        else:
            return 'Neutral'

    def extract_entities(self, title: str):
        key = title.lower()
        if key not in self.indices:
            return []
        idx = self.indices[key]
        text = self.df.loc[idx, 'content']
        doc = self.nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]

# Streamlit App
@st.cache_resource
def load_recommender(path: str):
    return BookRecommender(path)

def main():
    st.title("📚 Book Recommender System")
    # Sidebar inputs
    st.sidebar.header("Settings")
    data_path = st.sidebar.text_input("CSV file path", "dataset.csv")
    book_title = st.sidebar.text_input("Enter book title:")
    num_rec = st.sidebar.number_input("Number of recommendations", min_value=1, max_value=20, value=5)
    # Load or cache model
    recommender = load_recommender(data_path)

    if st.sidebar.button("Recommend Books"):
        recs = recommender.recommend(book_title, num_rec)
        if recs:
            st.subheader("Recommended Books:")
            for r in recs:
                st.write("- ", r)
        else:
            st.error("Book not found. Please check the title.")

    with st.expander("Book Sentiment Analysis"):
        sentiment = recommender.analyze_sentiment(book_title)
        if sentiment:
            st.write(f"Sentiment: **{sentiment}**")
        else:
            st.write("No sentiment data available.")

    with st.expander("Named Entity Recognition"):
        entities = recommender.extract_entities(book_title)
        if entities:
            for ent, label in entities:
                st.write(f"- {ent} ({label})")
        else:
            st.write("No entities found.")

if __name__ == '__main__':
    main()
