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
import streamlit as st

for pkg in ('stopwords','wordnet'):
    try:
        nltk.data.find(f'corpora/{pkg}')
    except LookupError:
        nltk.download(pkg)

class BookRecommender:
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self._prepare()

    def preprocess_text(self, text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            return ""
        text = unicodedata.normalize('NFKD', text)
        text = text.lower()
        text = re.sub(r"[^a-z]", " ", text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = WordPunctTokenizer().tokenize(text)
        tokens = [t for t in tokens if t not in self.stop_words]
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        tokens = [t for t in tokens if len(t) > 2]
        return ' '.join(tokens)

    def _prepare(self):
        self.df['content'] = self.df['Title'].fillna('') + ' ' + self.df['genres'].fillna('')
        self.df['Cleaned_Content'] = self.df['content'].apply(self.preprocess_text)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['Cleaned_Content'])
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
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

# Streamlit App
@st.cache_resource
def load_recommender(path: str):
    return BookRecommender(path)

def main():
    st.title("📚 Book Recommender System")
    st.sidebar.header("Settings")
    data_path = st.sidebar.text_input("CSV file path", "dataset.csv")
    book_title = st.sidebar.text_input("Enter book title:")
    num_rec = st.sidebar.number_input("Number of recommendations", min_value=1, max_value=20, value=5)
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

if __name__ == '__main__':
    main()
