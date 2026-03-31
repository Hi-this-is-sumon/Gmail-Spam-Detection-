from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pickle
import pandas as pd
import re
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import os

# Initialize App
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, including chrome-extension://
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model & Vectorizer
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "model/spam_model.pkl")
vectorizer_path = os.path.join(base_dir, "model/vectorizer.pkl")
whitelist_path = os.path.join(base_dir, "data/trusted_domains.csv")

model = None
vectorizer = None
trusted_domains = set()
ps = PorterStemmer()
STOPWORDS = set(ENGLISH_STOP_WORDS)


def load_resources():
    global model, vectorizer, trusted_domains

    try:
        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            with open(model_path, "rb") as model_file:
                model = pickle.load(model_file)
            with open(vectorizer_path, "rb") as vectorizer_file:
                vectorizer = pickle.load(vectorizer_file)
        else:
            print("Model or vectorizer not found. Please train the model first.")
    except Exception as exc:
        model = None
        vectorizer = None
        print(f"Error loading model resources: {exc}")

    try:
        if os.path.exists(whitelist_path):
            df = pd.read_csv(whitelist_path, header=None)
            if not df.empty:
                trusted_domains = set(df[0].astype(str).str.lower().values)
        else:
            print("Trusted domains file not found.")
    except Exception as exc:
        trusted_domains = set()
        print(f"Error loading trusted domains: {exc}")


load_resources()

class EmailRequest(BaseModel):
    sender: str
    subject: str
    body: str

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
        
    # Truncate to 3000 chars (matches training)
    text = text[:3000]
    
    # Match training script: allow numbers and $
    text = re.sub(r'[^a-z0-9\s$]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [ps.stem(word) for word in text if not word in STOPWORDS]
    text = ' '.join(text)
    return text

def check_whitelist(sender):
    if not trusted_domains:
        return False
    
    sender = sender.lower()
    domain = sender.split('@')[-1]
    
    if domain in trusted_domains:
        return True
    return False

def is_financial_or_transactional(sender, subject, body):
    """Check if email is from known financial/transactional services"""
    sender = sender.lower()
    
    # Known legitimate financial and service domains
    financial_domains = [
        'sbi.co.in', 'icicibank.com', 'hdfcbank.com', 'axisbank.com',
        'amazonpay.in', 'paytm.com', 'phonepe.com', 'googlepay.com',
        'paypal.com', 'razorpay.com', 'instamojo.com',
        'amazon.in', 'amazon.com', 'flipkart.com',
        'swiggy.com', 'zomato.com', 'uber.com', 'ola.com',
        'irctc.co.in', 'makemytrip.com', 'goibibo.com',
        'netflix.com', 'spotify.com', 'hotstar.com',
        'google.com', 'microsoft.com', 'apple.com'
    ]
    
    # Check sender domain
    domain = sender.split('@')[-1] if '@' in sender else ''
    for trusted_domain in financial_domains:
        if trusted_domain in domain:
            return True
    
    return False

def has_spam_indicators(subject, body):
    """
    Check for strong spam indicators that should override other checks.
    Returns True if email has obvious spam characteristics.
    """
    text = f"{subject} {body}".lower()
    
    # Strong spam indicators
    spam_keywords = [
        'winner', 'won', 'lottery', 'prize', 'claim now', 'urgent act',
        'limited time', 'click here now', 'congratulations!!!',
        'free money', 'million dollars', 'bitcoin', 'cryptocurrency',
        'act now', 'expire soon', 'verify your account immediately',
        'suspended account', 'confirm identity', 'wire transfer'
    ]
    
    # Count spam indicators
    spam_count = sum(1 for keyword in spam_keywords if keyword in text)
    
    # If 2+ strong spam indicators, definitely spam
    return spam_count >= 2

@app.get("/")
def home():
    return {"message": "Spam Detector API is running"}

@app.post("/predict")
def predict_spam(email: EmailRequest):
    if model is None or vectorizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Layer 1: Check Whitelist (CSV-based trusted domains)
    if check_whitelist(email.sender):
        return {
            "label": "whitelisted",
            "confidence": 1.0,
            "reason": "Sender is in the whitelist"
        }
    
    # Layer 2: Check for known financial/transactional services
    if is_financial_or_transactional(email.sender, email.subject, email.body):
        return {
            "label": "Not Spam",
            "confidence": 0.95,
            "reason": "Recognized as legitimate financial or transactional email"
        }
    
    # Layer 2.5: Check for obvious spam indicators (override ML if found)
    if has_spam_indicators(email.subject, email.body):
        return {
            "label": "Spam",
            "confidence": 0.95,
            "reason": "Contains multiple spam indicators"
        }

    # Layer 3: ML Model Prediction with Balanced Threshold
    full_text = f"{email.subject} {email.body}"
    processed_text = preprocess_text(full_text)
    vectorized_text = vectorizer.transform([processed_text]).toarray()
    
    prediction = model.predict(vectorized_text)[0]
    proba = model.predict_proba(vectorized_text)[0]
    
    # Balanced threshold: 50% for standard Naive Bayes
    spam_threshold = 0.5
    spam_probability = float(proba[1])
    ham_probability = float(proba[0])
    
    if spam_probability >= spam_threshold:
        label = "Spam"
        confidence = spam_probability
        analysis = f"AI Analysis: High probability of spam ({confidence:.1%}). Detected suspicious patterns typical of unsolicited emails."
    else:
        label = "Not Spam"
        confidence = ham_probability
        analysis = f"AI Analysis: Appears legitimate ({confidence:.1%}). Content aligns with normal communication patterns."
    
    reason = "Contains suspicious keywords and patterns" if label == "Spam" else "Appears to be legitimate"

    return {
        "label": label,
        "confidence": round(confidence, 2),
        "reason": reason,
        "analysis": analysis,
        "model_version": "MultinomialNB-v2 (Super Accurate)"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
