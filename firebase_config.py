import os
import json
import logging

import firebase_admin
from firebase_admin import credentials, storage, auth, firestore
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def initialize_firebase():
    """Initialize Firebase Admin SDK with service account credentials."""
    try:
        try:
            firebase_admin.get_app()
            logger.info("Firebase Admin SDK already initialized")
            return
        except ValueError:
            pass  # Not initialized yet, continue

        credentials_json = os.getenv("FIREBASE_CREDENTIALS")
        if not credentials_json:
            raise ValueError("Required environment variable 'FIREBASE_CREDENTIALS' is not set")

        storage_bucket = os.getenv("FIREBASE_STORAGE_BUCKET")
        if not storage_bucket:
            raise ValueError("Required environment variable 'FIREBASE_STORAGE_BUCKET' is not set")

        try:
            cred_dict = json.loads(credentials_json)
            cred = credentials.Certificate(cred_dict)
            logger.info("Firebase credentials loaded from JSON")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in FIREBASE_CREDENTIALS: {str(e)}")

        firebase_admin.initialize_app(cred, {"storageBucket": storage_bucket})
        logger.info(f"Firebase Admin SDK initialized successfully with bucket: {storage_bucket}")
    except Exception as e:
        logger.error(f"Failed to initialize Firebase: {str(e)}")
        raise


initialize_firebase()

db = firestore.client()
