# Book Recommender System

A Streamlit application that recommends books based on title similarity, performs sentiment analysis, and extracts named entities from book metadata.

## Features

- **Content-Based Recommendations**: Uses TF-IDF vectorization and cosine similarity to recommend similar books based on title and genre.
- **Sentiment Analysis**: Leverages TextBlob to gauge the overall sentiment of the book's combined title and genre text.
- **Named Entity Recognition**: Utilizes spaCy's `en_core_web_sm` model to extract entities from book metadata.
- **Interactive Web UI**: Built with Streamlit for an easy-to-use, real-time interface.

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/book-recommender-app.git
   cd book-recommender-app
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv venv
   # On macOS/Linux
   source venv/bin/activate
   # On Windows
   venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

## Usage

```bash
streamlit run book_recommender_app.py
```

- **CSV Path**: Specify the path to your CSV file (default: `Goodreads_books_with_genres.csv`).
- **Book Title**: Enter the book title for which you want recommendations.
- **Number of Recommendations**: Choose how many similar titles to retrieve.

Explore the app to see:
- A list of recommended book titles.
- Sentiment classification (Positive/Neutral/Negative) for the selected title.
- Named entities extracted from the book metadata.

## Project Structure

```
book-recommender-app/
├── book_recommender_app.py  # Streamlit application
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

## CSV Format

Your CSV should include at minimum the following columns:

- `Title`
- `genres`

Additional columns are ignored by the preprocessing pipeline.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- [NLTK](https://www.nltk.org/) for text preprocessing utilities.
- [spaCy](https://spacy.io/) for Named Entity Recognition.
- [TextBlob](https://textblob.readthedocs.io/) for sentiment analysis.
- [Streamlit](https://streamlit.io/) for UI components.

## Contact

For questions or suggestions, please open an issue or pull request on GitHub. 

