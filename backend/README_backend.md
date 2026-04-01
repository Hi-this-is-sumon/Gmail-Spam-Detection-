# Backend Documentation

This directory contains the Python backend for the Spam Detector Extension.

## Files

-   `app.py`: The FastAPI application server.
-   `requirements.txt`: Python dependencies.
-   `data/`: Contains datasets (`spam.csv`, `trusted_domains.csv`, `whitelist.csv`).
-   `model/`: Contains the training script (`train_model.py`) and saved models (`spam_model.pkl`, `vectorizer.pkl`).
-   `static/`: Static files for the web interface (CSS, JS, etc.).
-   `templates/`: HTML templates for the web interface.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Ensure the model is trained (see Training section below).

3. Run the server:
   ```bash
   python app.py
   ```
   The server will start on `http://127.0.0.1:8000`.

## API Endpoints

### `GET /`

Serves the landing page (index.html).

### `GET /health`

Health check endpoint.

**Response:**
```json
{
  "message": "Spam Detector API is running"
}
```

### `POST /predict`

**Request Body:**
```json
{
  "sender": "sender@example.com",
  "subject": "Subject line",
  "body": "Email body content..."
}
```

**Response:**
```json
{
  "label": "spam",
  "confidence": 0.95,
  "reason": "Contains suspicious keywords",
  "analysis": "AI Analysis: High probability of spam (95.0%). Detected suspicious patterns typical of unsolicited emails.",
  "model_version": "MultinomialNB-v2 (Super Accurate)"
}
```

## Training the Model

To retrain the model with new data:
1.  Update `data/spam.csv`.
2.  Run:
    ```bash
    python model/train_model.py
    ```
