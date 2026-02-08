import pandas as pd
import nltk
from .config import Config

class DataLoader:
    def __init__(self):
        # Ensure NLTK data is available
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            nltk.download('punkt_tab')

    def load_data(self, posts_path=None, comments_path=None, nrows=None):
        """
        Loads posts and comments, merges them, and returns a DataFrame.
        """
        posts = pd.DataFrame()
        comments = pd.DataFrame()

        if posts_path and os.path.exists(posts_path):
            print(f"Loading posts from {posts_path}...")
            posts = pd.read_csv(posts_path, nrows=nrows)
            # Create a unified text column
            posts[Config.TEXT_COLUMN] = posts['title'].fillna('') + " " + posts['body'].fillna('')
        
        if comments_path and os.path.exists(comments_path):
             print(f"Loading comments from {comments_path}...")
             comments = pd.read_csv(comments_path, nrows=nrows)
             # Comments usually just have 'body' or 'comment'
             if 'body' in comments.columns:
                 comments[Config.TEXT_COLUMN] = comments['body'].fillna('')
             elif 'comment' in comments.columns:
                 comments[Config.TEXT_COLUMN] = comments['comment'].fillna('')

        # Add source type for tracking
        if not posts.empty:
            posts['source_type'] = 'post'
        if not comments.empty:
            comments['source_type'] = 'comment'
            
        # Combine
        full_df = pd.concat([posts, comments], ignore_index=True)
        
        # Ensure date is datetime
        # Iterate through possible date columns
        date_cols = ['created_utc', 'date', 'timestamp', 'created']
        found_date = False
        for col in date_cols:
            if col in full_df.columns:
                full_df[Config.DATE_COLUMN] = pd.to_datetime(full_df[col], unit='s' if col == 'created_utc' else None, errors='coerce')
                found_date = True
                break
        
        if not found_date:
            print("Warning: No suitable date column found. Time series analysis might fail.")

        return full_df

    def preprocess_sentences(self, df):
        """
        Splits text into sentences. Returns a list of (sentence, date, source_id) tuples.
        """
        print("Tokenizing sentences...")
        sentences_data = []
        for index, row in df.iterrows():
            text = str(row.get(Config.TEXT_COLUMN, ''))
            date = row.get(Config.DATE_COLUMN, pd.NaT)
            
            if not text.strip():
                continue
                
            raw_sentences = nltk.sent_tokenize(text)
            for s in raw_sentences:
                sentences_data.append({
                    'sentence': s,
                    'date': date,
                    'original_index': index
                })
                
        return pd.DataFrame(sentences_data)
import os 
