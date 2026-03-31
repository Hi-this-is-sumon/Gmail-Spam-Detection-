
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os

# Ensure NLTK data
nltk.download('stopwords')
ps = PorterStemmer()

STOPWORDS = set(stopwords.words('english'))

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

def verify():
    print("Loading model...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'model', 'spam_model.pkl')
    vectorizer_path = os.path.join(base_dir, 'model', 'vectorizer.pkl')

    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}!")
        return
    
    model = pickle.load(open(model_path, 'rb'))
    vectorizer = pickle.load(open(vectorizer_path, 'rb'))
    
    test_cases = [
        # Obvious Spam
        ("WINNER!! As a valued network customer you have been selected to receive a £900 prize reward!", "SPAM"),
        ("Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005", "SPAM"),
        
        # Obvious Ham
        ("Hey, are we still meeting for lunch today?", "HAM"),
        ("Your Amazon order #123-456 has been shipped.", "HAM"),
        
        # Difficult / Subtle Spam (Phishing/Fraud)
        ("URGENT: Your account has been suspended. Click here to verify.", "SPAM"),
        ("Dear customer, please review the attached invoice #998877 for payment immediately.", "SPAM"),
        ("Security Alert: We noticed a new login from Russia. Reset your password now.", "SPAM"),
        
        # Tricky Ham (Legitimate but sounds spammy)
        ("Congratulations! You have been promoted to Senior Developer.", "HAM"),
        ("Here is the free report you requested on Q3 earnings.", "HAM"),
        ("Limited time offer: 50% off on all shoes at our store this weekend only.", "HAM") # Marketing email, usually HAM
    ]
    
    print(f"\nRunning {len(test_cases)} test cases...")
    passed = 0
    for text, expected in test_cases:
        # Predict
        processed_text = preprocess_text(text)
        vectorized_text = vectorizer.transform([processed_text]).toarray()
        prediction = model.predict(vectorized_text)[0]
        proba = model.predict_proba(vectorized_text)[0]
        
        # Threshold logic (same as app.py)
        spam_threshold = 0.5
        if proba[1] >= spam_threshold:
            label = "SPAM"
            confidence = proba[1]
        else:
            label = "HAM"
            confidence = proba[0]
            
        # Map to UI labels for checking
        ui_label = "Spam" if label == "SPAM" else "Not Spam"
        expected_ui = "Spam" if expected == "SPAM" else "Not Spam"
            
        result = "PASS" if ui_label == expected_ui else "FAIL"
        print(f"Text: {text[:50]}...")
        print(f"Expected: {expected_ui}, Got: {ui_label} (Conf: {confidence:.2f}) - {result}")
        print("-" * 30)
        
        if result == "PASS":
            passed += 1
        
    print(f"\nAccuracy on test cases: {passed}/{len(test_cases)} ({passed/len(test_cases)*100:.1f}%)")

if __name__ == "__main__":
    verify()
