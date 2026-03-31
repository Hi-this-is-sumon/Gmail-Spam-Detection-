# Gmail-Spam-Detection-

AI-powered spam detection system for Gmail with a modern, animated Chrome extension and Python backend.

**Live API:** `https://gmail-spam-detection-nine.vercel.app/`

**Last Updated:** March 31, 2026

## Features

- **Advanced Animated SVG Icon:** Futuristic hexagonal shield with pulsing, rotating layers and orbiting particles
- **Modern UI:** Vibrant neon colors, breathing animations, floating particles
- **Clear Labels:** "Spam" / "Not Spam" / "Whitelisted" for easy understanding
- **Single Input:** Paste any email for instant analysis
- **4-Layer Detection:** Whitelist, financial domains, spam keywords, ML model
- **50% Accuracy Threshold:** Balanced spam detection (MultinomialNB)
- **Real-time Gmail Integration:** Auto-scan opened emails
- **Transactional Email Recognition:** Detects legitimate financial services

## Tech Stack

### Frontend (Chrome Extension)
- Manifest V3
- Modern CSS animations (50+ keyframes)
- Glassmorphism design
- Responsive UI

### Backend (Python)
- FastAPI
- scikit-learn (MultinomialNB)
- NLTK for text processing (stemming, stopwords)
- TF-IDF vectorization (5000 features, bigrams)
- Class balancing via oversampling

## Quick Start for Users

Normal users do **not** need to run Python locally. The backend is already deployed on Vercel.

### Install the Chrome Extension

1. Open the GitHub repository: `https://github.com/Hi-this-is-sumon/Gmail-Spam-Detection-`
2. Click **Code** → **Download ZIP**
3. Extract the ZIP file on your computer
4. Open Chrome and go to `chrome://extensions/`
5. Turn on **Developer mode**
6. Click **Load unpacked**
7. Select the `extension` folder from the extracted project
8. Pin the extension and start using it in Gmail

### How to Use It

1. Open Gmail and select any email
2. Click the extension icon
3. Press **Get from Gmail** or paste email text manually
4. Click **Analyze** to see the spam result

> No login or registration is required for this version.

## Installation for Developers

### Clone from GitHub

```bash
git clone https://github.com/Hi-this-is-sumon/Gmail-Spam-Detection-.git
cd Gmail-Spam-Detection-
```

### Backend Setup

1. Navigate to backend directory:
```bash
cd backend
```

2. Create virtual environment:
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Train the model:
```bash
python model/train_model.py
```

5. Start the server:
```bash
python app.py
```

Server will run on `http://127.0.0.1:8000`

### Extension Setup

1. Open Chrome and go to `chrome://extensions/`
2. Enable "Developer mode"
3. Click "Load unpacked"
4. Select the `extension` folder
5. Pin the extension to your toolbar
6. Reload the extension after any code update

## Usage

1. **Manual Analysis:**
   - Click extension icon
   - Paste complete email text
   - Click "Analyze"

2. **Gmail Integration:**
   - Open any email in Gmail
   - Click "Get from Gmail" in extension
   - Click "Analyze"

3. **Auto-Detection:**
   - Content script automatically scans opened emails
   - Red banner appears if spam detected

## Project Structure

```
Gmail-Spam-Detection-/
├── backend/
│   ├── data/
│   │   ├── spam.csv           # Training dataset
│   │   └── trusted_domains.csv # Whitelist
│   ├── model/
│   │   ├── train_model.py     # ML model training
│   │   ├── spam_model.pkl     # Trained model
│   │   └── vectorizer.pkl     # TF-IDF vectorizer
│   ├── app.py                 # FastAPI server
│   ├── requirements.txt       # Python dependencies
│   └── verify_model.py        # Model testing
├── extension/
│   ├── assets/                # Icons
│   ├── manifest.json          # Extension config
│   ├── popup.html            # Popup UI
│   ├── popup.css             # Styles & animations
│   ├── popup.js              # Popup logic
│   ├── content.js            # Gmail integration
│   └── background.js         # Service worker
└── README.md
```

## Detection Layers

1. **Whitelist Check:** Custom trusted domains
2. **Financial Domain Verification:** 30+ known services (SBI, Amazon Pay, etc.)
3. **Spam Keyword Detection:** 15+ spam indicators
4. **ML Model:** MultinomialNB with 50% confidence threshold (0.5)

## Customization

### Add Trusted Domains
Edit `backend/data/trusted_domains.csv` - one domain per line

### Adjust Threshold
In `backend/app.py`, change `spam_threshold` value (currently 0.75)

### Retrain Model
Add emails to `backend/data/spam.csv` and run:
```bash
python model/train_model.py
```

## Performance

- **99.88% Training Accuracy** (after class balancing)
- **70%+ Verification Accuracy** on difficult test cases
- **50% Production Threshold** (MultinomialNB, optimized for balance)
- **Multi-layer filtering** reduces false positives
- **Recognizes 30+ financial services**

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## License

This project is licensed under the **MIT License**. See the [`LICENSE`](LICENSE) file for full details.

## Author

**Sumon Mandal**   
GitHub: [@Hi-this-is-sumon](https://github.com/Hi-this-is-sumon)

Built as a final project for Secure & Intelligent Computing (SIC) course.

---

*Project developed with focus on real-world spam detection using machine learning and modern web technologies.*
