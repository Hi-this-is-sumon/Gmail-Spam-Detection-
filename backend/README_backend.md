# Backend Documentation

This directory contains the Python backend for the Spam Detector Extension.

## Files

-   `app.py`: The FastAPI application server.
-   `requirements.txt`: Python dependencies.
-   `data/`: Contains datasets (`spam.csv`, `whitelist.csv`).
-   `model/`: Contains the training script and saved models.

## API Endpoints

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
  "reason": "Contains suspicious keywords"
}
```

## Training the Model

To retrain the model with new data:
1.  Update `data/spam.csv`.
2.  Run:
    ```bash
    python model/train_model.py
    ```
