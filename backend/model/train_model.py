import pandas as pd
import pickle
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os

# Download NLTK data
nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    
    # Truncate to 3000 chars to prevent hanging on large emails
    text = text[:3000]
    
    # Lowercase
    text = text.lower()
    
    # Keep only letters and some punctuation
    text = re.sub(r'[^a-z0-9\s$]', ' ', text)
    
    # Split
    text = text.split()
    
    # Stemming
    ps = PorterStemmer()
    text = [ps.stem(word) for word in text if word not in STOPWORDS]
    
    text = ' '.join(text)
    return text

def train():
    print("DEBUG: Running optimized train_model.py")
    # Get script directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, '..', 'data', 'spam.csv')
    model_path = os.path.join(base_dir, 'spam_model.pkl')
    vectorizer_path = os.path.join(base_dir, 'vectorizer.pkl')

    print(f"Loading dataset from {data_path}...")
    try:
        df = pd.read_csv(data_path, encoding='latin-1')
    except FileNotFoundError:
        print(f"Error: spam.csv not found at {data_path}")
        return

    if 'Body' in df.columns and 'Label' in df.columns:
        df = df[['Body', 'Label']]
        df.columns = ['message', 'label']
    elif 'v1' in df.columns and 'v2' in df.columns:
        df = df[['v2', 'v1']]
        df.columns = ['message', 'label']
        # Map label to 0/1 if it's ham/spam string
        df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    
    print(f"Dataset shape: {df.shape}")
    print(f"Class distribution before balancing:\n{df['label'].value_counts()}")

    # Handle Imbalance: Oversample Spam
    spam_df = df[df['label'] == 1]
    ham_df = df[df['label'] == 0]
    
    # Oversample spam to match ham count
    spam_upsampled = spam_df.sample(n=len(ham_df), replace=True, random_state=42)
    df_balanced = pd.concat([ham_df, spam_upsampled])
    
    print(f"Class distribution after balancing:\n{df_balanced['label'].value_counts()}")

    print("Preprocessing data...")
    df_balanced.dropna(inplace=True)
    df_balanced['message'] = df_balanced['message'].apply(preprocess_text)
    df_balanced['label'] = df_balanced['label'].astype(int)

    X = df_balanced['message']
    y = df_balanced['label']

    # Vectorization with N-grams
    print("Vectorizing text with N-grams...")
    cv = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = cv.fit_transform(X).toarray()

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # Train with MultinomialNB (Best for text)
    print("Training MultinomialNB model...")
    from sklearn.naive_bayes import MultinomialNB
    
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save
    print(f"Saving model to {model_path}...")
    pickle.dump(model, open(model_path, 'wb'))
    pickle.dump(cv, open(vectorizer_path, 'wb'))
    print("Done!")

if __name__ == "__main__":
    train()
