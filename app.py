import os
import json
import io
import tempfile
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from flask import Flask, jsonify, request, make_response, redirect
from werkzeug.exceptions import BadRequest
from flask_cors import CORS
from dotenv import load_dotenv
from pinecone import (Pinecone)
from pinecone.grpc import PineconeGRPC
import logging
import firebase_admin
from firebase_admin import credentials, storage, auth, firestore
from openai import OpenAI
import PyPDF2
from docx import Document
from firecrawl import Firecrawl
import pandas as pd
import duckdb
import json
import re
import difflib
from typing import Any, Dict, List, Tuple, Optional
from functools import wraps
from urllib.parse import urlencode
import requests
import secrets
import uuid
from zoneinfo import ZoneInfo
from bs4 import BeautifulSoup, NavigableString, Tag
import html
import stripe
import urllib.parse

# from typing import List
# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Get allowed origins from environment variable (comma-separated)
# Default to localhost and production frontend for backward compatibility
allowed_origins_str = os.getenv('ALLOWED_ORIGINS', 'http://localhost:3000,https://notebook-mvp.vercel.app')
allowed_origins = [origin.strip() for origin in allowed_origins_str.split(',') if origin.strip()]

# Configure CORS with more specific settings
CORS(app, 
     resources={r"/*": {
         "origins": allowed_origins,
         "methods": ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
         "allow_headers": ["Content-Type", "Authorization", "X-Requested-With", "x-solari-key"],
         "expose_headers": ["Content-Type", "Content-Disposition"],
         "supports_credentials": True,
         "max_age": 86400
     }})

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Initialize Firecrawl with error handling
firecrawl_api_key = os.getenv('FIRECRAWL_API_KEY')
if firecrawl_api_key:
    try:
        firecrawl = Firecrawl(api_key=firecrawl_api_key)
        logger.info("Firecrawl initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize Firecrawl: {str(e)}. Firecrawl features will be unavailable.")
        firecrawl = None
else:
    logger.warning("FIRECRAWL_API_KEY not set, Firecrawl features will be unavailable")
    firecrawl = None


# Helper functions to load environment variables
def get_env_var(key: str, default: str = None, required: bool = False) -> str:
    """
    Get an environment variable from .env file.
    
    Args:
        key: The environment variable key
        default: Default value if key is not found (optional)
        required: If True, raises an error if key is not found
    
    Returns:
        The environment variable value as a string
    
    Raises:
        ValueError: If required=True and key is not found
    """
    value = os.getenv(key, default)
    
    if required and value is None:
        raise ValueError(f"Required environment variable '{key}' is not set")
    
    return value

def get_api_key(key_name: str, required: bool = True) -> str:
    """
    Get an API key from environment variables.
    
    Args:
        key_name: The name of the API key (e.g., 'OPENAI_API_KEY')
        required: If True, raises an error if key is not found
    
    Returns:
        The API key value as a string
    
    Raises:
        ValueError: If required=True and key is not found
    """
    return get_env_var(key_name, required=required)

def get_all_env_vars() -> dict:
    """
    Get all environment variables as a dictionary.
    
    Returns:
        Dictionary of all environment variables
    """
    return dict(os.environ)

# Jira OAuth Configuration
ATLASSIAN_REDIRECT_URI = "https://api.usesolari.ai/auth/jira/callback"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
FRONTEND_SUCCESS_URL = os.getenv('FRONTEND_SUCCESS_URL', 'http://localhost:3000')

# Security: Solari Key validation
SOLARI_INTERNAL_KEY = os.environ.get("SOLARI_INTERNAL_KEY")
SOLARI_DEV_KEY = os.environ.get("SOLARI_DEV_KEY")
SOLARI_KEY_HEADER = "x-solari-key"
def utcnow():
    return datetime.now(timezone.utc)

def _is_valid_solari_key(key: str | None) -> bool:
    if not key:
        return False
    # Accept either key (dev key is for curl / personal testing)
    if SOLARI_INTERNAL_KEY and key == SOLARI_INTERNAL_KEY:
        return True
    if SOLARI_DEV_KEY and key == SOLARI_DEV_KEY:
        return True
    return False

def require_solari_key(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        # If neither key is set, we don't enforce (useful for local dev).
        # In production, you should always set SOLARI_INTERNAL_KEY in Render.
        if not SOLARI_INTERNAL_KEY and not SOLARI_DEV_KEY:
            return fn(*args, **kwargs)
        key = request.headers.get(SOLARI_KEY_HEADER)
        if not _is_valid_solari_key(key):
            return jsonify({"error": "unauthorized"}), 401
        return fn(*args, **kwargs)
    return wrapper

# Custom CORS Headers Helper Function
def add_cors_headers(response):
    """Helper function to add CORS headers to any response"""
    origin = request.headers.get('Origin')
    # Only allow origins from the configured whitelist
    if origin and origin in allowed_origins:
        response.headers['Access-Control-Allow-Origin'] = origin
    elif allowed_origins:
        # If origin is not in whitelist, use first allowed origin as fallback
        # This prevents wildcard '*' which is insecure with credentials
        response.headers['Access-Control-Allow-Origin'] = allowed_origins[0]
    else:
        # Fallback if no origins configured (shouldn't happen in production)
        response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, PATCH, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With, x-solari-key'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    response.headers['Access-Control-Expose-Headers'] = 'Content-Type, Content-Disposition'
    response.headers['Access-Control-Max-Age'] = '86400'
    return response

def send_email(to_email, subject, body, html_body=None):
    """Sends an email using Mailgun API"""
    mailgun_api_key = os.getenv('MAILGUN_API_KEY')
    if not mailgun_api_key:
        logger.error("Mailgun API key not found in environment variables")
        return None

    payload = {
        "from": "Sahil's Robots @ Solari <postmaster@robots.usesolari.ai>",
        "to": to_email,
        "subject": subject,
        "text": body,
    }
    if html_body:
        payload["html"] = html_body

    try:
        response = requests.post(
            "https://api.mailgun.net/v3/robots.usesolari.ai/messages",
            auth=("api", mailgun_api_key),
            data=payload,
            timeout=10
        )
        return response
    except Exception as e:
        logger.error(f"Error sending email: {str(e)}")
        return None

def build_signup_welcome_email(user_data: dict):
    display_name = (user_data.get("displayName") or user_data.get("name") or "").strip()
    first_name = display_name.split()[0] if display_name else ""
    greeting = f"Hey {first_name}!" if first_name else "Hey!"

    calendar_url = os.getenv(
        "WELCOME_CALENDAR_URL",
        "https://cal.com/sahil-sinha-hugr4z/solari-onboarding",
    )
    docs_url = "https://docs.usesolari.ai"

    subject = "Welcome to Solari 👋"
    text_body = (
        f"{greeting}\n\n"
        "Welcome to Solari! I'm Sahil, I'm the CEO and Founder of Solari AI. "
        "I'd love to help you and your team set up your Solari account for success. "
        "If you've got the time, I'd love to onboard you personally, and get to know "
        "more about you and your team! You can always grab time on my calendar here: "
        f"{calendar_url}\n\n"
        "(No worries if you can't decide yet — I'll follow up in a few days as well.)\n\n"
        f"You can also get set up yourself using our docs ({docs_url}). "
        "(If you're looking for somewhere to start, I'd recommend creating your own "
        "Chat (RAG) Agent, and connecting either a Slack channel or a website as a "
        "source, and start playing around with Solari!\n\n)"
        "If you have any questions, need some last mile configurations, or need help "
        "thinking through what agents could help your team — I'd love to chat: "
        f"{calendar_url}\n\n"
        "Cheers,\n"
        "Sahil"
    )
    create_agent_url = "https://docs.usesolari.ai/essentials/create-agent#create-agent"
    slack_connect_url = "https://docs.usesolari.ai/guides/getting-started-in-solari/integrations/slack"
    slack_channels_url = "https://docs.usesolari.ai/essentials/add_sources#slack-channels"
    website_sources_url = "https://docs.usesolari.ai/essentials/add_sources#websites"
    run_query_url = "https://docs.usesolari.ai/guides/running-your-agents/run-query"

    html_body = (
        f"{greeting}<br><br>"
        "Welcome to Solari! I'm Sahil, I'm the CEO and Founder of Solari AI. "
        "I'd love to help you and your team set up your Solari account for success. "
        "If you've got the time, I'd love to onboard you personally, and get to know "
        "more about you and your team! You can always grab time on my calendar here: "
        f'<a href="{calendar_url}">{calendar_url}</a><br><br>'
        "(No worries if you can't decide yet — I'll follow up in a few days as well.)<br><br>"
        f'You can also get set up yourself using our <a href="{docs_url}">docs</a>. '
        "If you're looking for somewhere to start, I'd recommend "
        f'<a href="{create_agent_url}">creating your own Chat (RAG) Agent</a>, and '
        f'<a href="{slack_connect_url}">connecting</a> either a '
        f'<a href="{slack_channels_url}">Slack channel</a> or a '
        f'<a href="{website_sources_url}">website</a> as a source, and start '
        f'<a href="{run_query_url}">playing around with Solari</a>!<br><br>'
        "If you have any questions, need some last mile configurations, or need help "
        "thinking through what agents could help your team — I'd love to chat: "
        f'<a href="{calendar_url}">{calendar_url}</a><br><br>'
        "Cheers,<br>"
        "Sahil"
    )
    return subject, text_body, html_body

def update_job(
    job_ref,
    *,
    status=None,
    progress=None,
    message=None,
    locked_by=None,
    locked_until=None,
):
    updates = {}

    if status is not None:
        updates["status"] = status

    if progress is not None:
        updates["progress"] = progress

    if message is not None:
        updates["message"] = message

    if locked_by is not None:
        updates["locked_by"] = locked_by

    if locked_until is not None:
        updates["locked_until"] = locked_until

    if not updates:
        return

    updates["updated_at_unix"] = int(time.time())

    job_ref.update(updates)

def update_source(job_ref, source_key, patch):
    snap = job_ref.get()
    if not snap.exists:
        return

    job = snap.to_dict() or {}
    sources = job.get("sources", [])

    updated = False
    for i, src in enumerate(sources):
        if src.get("source_key") == source_key:
            new_src = dict(src)
            new_src.update(patch)
            sources[i] = new_src
            updated = True
            break

    if not updated:
        return

    job_ref.update({
        "sources": sources,
        "updated_at_unix": int(time.time()),
    })

# Global Response Handler
@app.after_request
def after_request(response):
    """Add CORS headers to every response"""
    return add_cors_headers(response)

# Preflight Request Handler
@app.before_request
def handle_preflight():
    """Global OPTIONS handler for all routes"""
    if request.method == "OPTIONS":
        response = make_response()
        return add_cors_headers(response), 204
    return None

# Initialize Firebase Admin SDK
def initialize_firebase():
    """Initialize Firebase Admin SDK with service account credentials"""
    try:
        # Check if Firebase is already initialized
        try:
            firebase_admin.get_app()
            logger.info("Firebase Admin SDK already initialized")
            return
        except ValueError:
            pass  # Not initialized yet, continue
        
        # Get credentials and storage bucket from environment
        credentials_json = get_env_var('FIREBASE_CREDENTIALS', required=True)
        storage_bucket = get_env_var('FIREBASE_STORAGE_BUCKET', required=True)
        
        # Parse JSON credentials
        try:
            cred_dict = json.loads(credentials_json)
            cred = credentials.Certificate(cred_dict)
            logger.info("Firebase credentials loaded from JSON")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in FIREBASE_CREDENTIALS: {str(e)}")
        
        # Initialize Firebase app with storage bucket
        firebase_admin.initialize_app(cred, {
            'storageBucket': storage_bucket
        })
        
        logger.info(f"Firebase Admin SDK initialized successfully with bucket: {storage_bucket}")
    except Exception as e:
        logger.error(f"Failed to initialize Firebase: {str(e)}")
        raise

# Initialize Firebase on startup
initialize_firebase()

# Firebase connection test function
def test_firebase_connection():
    """
    Test Firebase connection for Auth, Storage, and Firestore.
    
    Returns:
        dict: Status of each service with success/error information
    """
    results = {
        'auth': {'status': 'unknown', 'message': ''},
        'storage': {'status': 'unknown', 'message': ''},
        'firestore': {'status': 'unknown', 'message': ''}
    }
    
    # Test Auth
    try:
        # Try to list users (limited to 1 to just test connection)
        auth.list_users(max_results=1)
        results['auth'] = {
            'status': 'success',
            'message': 'Auth service is accessible'
        }
    except Exception as e:
        results['auth'] = {
            'status': 'error',
            'message': f'Auth connection failed: {str(e)}'
        }
    
    # Test Storage
    try:
        bucket = storage.bucket()
        # Try to list blobs (limited to 1 to just test connection)
        list(bucket.list_blobs(max_results=1))
        results['storage'] = {
            'status': 'success',
            'message': f'Storage bucket "{bucket.name}" is accessible'
        }
    except Exception as e:
        results['storage'] = {
            'status': 'error',
            'message': f'Storage connection failed: {str(e)}'
        }
    
    # Test Firestore
    try:
        db = firestore.client()
        # Try to get a collection reference (doesn't actually query, just tests connection)
        test_collection = db.collection('_test_connection')
        # Try to read (will succeed even if collection doesn't exist)
        list(test_collection.limit(1).stream())
        results['firestore'] = {
            'status': 'success',
            'message': 'Firestore database is accessible'
        }
    except Exception as e:
        results['firestore'] = {
            'status': 'error',
            'message': f'Firestore connection failed: {str(e)}'
        }
    
    return results

# set up firebase client
db = firestore.client()

# Initialize Pinecone client
def get_pinecone_client():
    """Initialize and return Pinecone client"""
    api_key = get_api_key('PINECONE_API_KEY')
    return Pinecone(api_key=api_key)

# Initialize PineconeGRPC client and index
def get_pinecone_grpc_index():
    """Initialize and return PineconeGRPC index using host"""
    api_key = get_api_key('PINECONE_API_KEY')
    index_host = get_env_var('PINECONE_INDEX_HOST', required=True)
    pc = PineconeGRPC(api_key=api_key)
    return pc.Index(host=index_host)

# Initialize OpenAI client
def get_openai_client():
    """Initialize and return OpenAI client"""
    # api_key = get_api_key('OPENAI_API_KEY')
    api_key = get_api_key('KEYWORDS_API_KEY_PROD')
    return OpenAI(base_url="https://api.keywordsai.co/api/", api_key=api_key)

def get_keywordsai_client() -> OpenAI:
    """Initialize and return KeywordsAI OpenAI-compatible client"""
    api_key = get_api_key('KEYWORDS_API_KEY_PROD')
    return OpenAI(base_url="https://api.keywordsai.co/api/", api_key=api_key)

def get_agent_model_provider(db, team_id: str, agent_id: str) -> str:
    """Fetch model_provider for an agent, defaulting if missing."""
    try:
        agent_ref = (
            db.collection("teams").document(team_id)
              .collection("agents").document(agent_id)
        )
        agent_snap = agent_ref.get()
        if not agent_snap.exists:
            return "gpt-4o-mini"
        agent_data = agent_snap.to_dict() or {}
        return agent_data.get("model_provider") or "gpt-4o-mini"
    except Exception:
        return "gpt-4o-mini"

def keywordsai_chat_completion(
    messages: list,
    model: str = "gpt-4o-mini",
    user_id: Optional[str] = None,
    temperature: Optional[float] = None,
) -> dict:
    """
    Call KeywordsAI chat completions with optional analytics metadata.
    
    Args:
        messages: OpenAI-style messages payload
        model: Model name for KeywordsAI router
        user_id: Optional user id for analytics
        temperature: Optional temperature override
    
    Returns:
        Parsed JSON response from KeywordsAI
    """
    try:
        client = get_keywordsai_client()
        extra_body = {}
        metadata = {}
        if user_id:
            extra_body["customer_identifier"] = user_id
            metadata["user_id"] = user_id
        if metadata:
            extra_body["metadata"] = metadata

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            extra_body=extra_body if extra_body else None
        )
        return response.model_dump()
    except Exception as e:
        logger.error(f"KeywordsAI request failed: {str(e)}")
        raise ValueError(f"KeywordsAI request failed: {str(e)}")

# File processing functions
def download_file_from_firebase(file_path: str) -> bytes:
    """
    Download file from Firebase Storage.
    
    Args:
        file_path: Firebase Storage path (e.g., 'users/userId/files/file.pdf')
    
    Returns:
        File content as bytes
    """
    try:
        bucket = storage.bucket()
        blob = bucket.blob(file_path)
        
        if not blob.exists():
            raise ValueError(f"File not found in Firebase Storage: {file_path}")
        
        file_content = blob.download_as_bytes()
        logger.info(f"Downloaded file from Firebase: {file_path} ({len(file_content)} bytes)")
        return file_content
    except Exception as e:
        logger.error(f"Error downloading file from Firebase: {str(e)}")
        raise ValueError(f"Failed to download file from Firebase: {str(e)}")

def extract_text_from_pdf(file_content: bytes) -> str:
    """Extract text from PDF file"""
    try:
        pdf_file = io.BytesIO(file_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        raise ValueError(f"Failed to extract text from PDF: {str(e)}")

def extract_text_from_docx(file_content: bytes) -> str:
    """Extract text from DOCX file"""
    try:
        docx_file = io.BytesIO(file_content)
        doc = Document(docx_file)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text
    except Exception as e:
        logger.error(f"Error extracting text from DOCX: {str(e)}")
        raise ValueError(f"Failed to extract text from DOCX: {str(e)}")

def extract_text_from_file(file_content: bytes, file_path: str) -> str:
    """Extract text from file based on file extension"""
    file_ext = file_path.lower().split('.')[-1]
    
    if file_ext == 'pdf':
        return extract_text_from_pdf(file_content)
    elif file_ext in ['doc', 'docx']:
        return extract_text_from_docx(file_content)
    elif file_ext == 'txt':
        return file_content.decode('utf-8')
    else:
        raise ValueError(f"Unsupported file type: {file_ext}")

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list:
    """
    Split text into chunks with overlap.
    
    Args:
        text: The text to chunk
        chunk_size: Size of each chunk in characters
        chunk_overlap: Number of characters to overlap between chunks
    
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to break at sentence boundary if not at end
        if end < len(text):
            # Look for sentence endings
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            break_point = max(last_period, last_newline)
            
            if break_point > chunk_size * 0.5:  # Only break if we're not too far from start
                chunk = chunk[:break_point + 1]
                end = start + break_point + 1
        
        chunks.append(chunk.strip())
        start = end - chunk_overlap
    
    return chunks

def generate_embeddings(text_chunks: list, openai_client: OpenAI, model: str = "text-embedding-3-small") -> list:
    """
    Generate embeddings for text chunks using OpenAI.
    
    Args:
        text_chunks: List of text chunks
        openai_client: OpenAI client instance
        model: Embedding model to use
    
    Returns:
        List of embedding vectors
    """
    try:
        response = openai_client.embeddings.create(
            model=model,
            input=text_chunks,
            dimensions=1024  # Match your Pinecone index dimension
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise ValueError(f"Failed to generate embeddings: {str(e)}")

def upload_vectors_to_pinecone(vectors: list, namespace: str, index_name: str = 'production', batch_size: int = 100) -> int:
    """
    Upload vectors to Pinecone in batches to avoid message size limits.
    
    Pinecone has a ~4MB limit per request. This function splits large uploads
    into smaller batches to stay under the limit.
    
    Args:
        vectors: List of vectors to upload
        namespace: Pinecone namespace
        index_name: Name of the Pinecone index
        batch_size: Number of vectors per batch (default 100, adjust based on vector size)
    
    Returns:
        Total number of vectors uploaded
    """
    if not vectors:
        return 0
    
    pc = get_pinecone_client()
    index = pc.Index(index_name)
    
    total_uploaded = 0
    num_batches = (len(vectors) + batch_size - 1) // batch_size  # Ceiling division
    
    # Upload in batches
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        batch_num = i // batch_size + 1
        logger.info(f"Uploading batch {batch_num}/{num_batches} ({len(batch)} vectors)...")
        index.upsert(vectors=batch, namespace=namespace)
        total_uploaded += len(batch)
    
    logger.info(f"Successfully uploaded {total_uploaded} vectors in {num_batches} batches")
    return total_uploaded

# Flask routes
@app.route('/', methods=['GET'])
@require_solari_key
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Flask API is running'
    }), 200

@app.route('/health', methods=['GET'])
def health():
    """Simple health endpoint for testing (no authentication required)"""
    return jsonify({"status": "ok"}), 200

@app.route('/api/status', methods=['GET'])
@require_solari_key
def status():
    """API status endpoint"""
    return jsonify({
        'status': 'success',
        'message': 'API is operational'
    }), 200

@app.route("/auth/jira/connect", methods=['GET'])
def jira_connect():
    """
    Step 1: Redirect user to Atlassian OAuth screen to approve access.
    
    Usage: GET /auth/jira/connect?uid={user_id}&client_id={jira_client_id}
    """
    uid = request.args.get("uid")
    client_id = request.args.get("client_id") or os.getenv("JIRA_CLIENT_ID")
    
    if not uid:
        return jsonify({"error": "Missing uid parameter"}), 400
    
    if not client_id:
        return jsonify({"error": "Jira client_id not configured"}), 500
    
    query_params = {
    "audience": "api.atlassian.com",
    "client_id": client_id,
    "scope": (
        "offline_access "
        "read:jira-work "
        "read:confluence-space.summary "
        "read:confluence-props "
        "read:confluence-content.all "
        "read:confluence-content.summary "
        "search:confluence "
        "read:space:confluence "
        "read:page:confluence"
    ),
    "redirect_uri": ATLASSIAN_REDIRECT_URI,
    "response_type": "code",
    "prompt": "consent",
    "state": uid,
        }
    
    auth_url = "https://auth.atlassian.com/authorize?" + urlencode(query_params)
    return redirect(auth_url)

# BILLING WORK

# get stripe vars
stripe.api_key = os.environ["STRIPE_SECRET_KEY"]
STRIPE_PRICE_ID = os.environ["STRIPE_PRICE_ID"]
STRIPE_PRICE_ID_DISCOUNT = os.environ.get("STRIPE_PRICE_ID_DISCOUNT")

# build frontend urls
APP_URL = os.environ.get("APP_URL", "http://localhost:3000")
SUCCESS_URL = f"{APP_URL}/billing/success?session_id={{CHECKOUT_SESSION_ID}}"
CANCEL_URL = f"{APP_URL}/billing/cancel"


@app.route("/api/stripe/create_checkout_session", methods=["POST"])
def create_checkout_session():
    data = request.get_json(silent=True) or {}
    user_id = data.get("user_id")
    purchase = bool(data.get("purchase"))

    if not user_id:
        return jsonify({"error": "Missing user_id"}), 400

    user_snap = db.collection("users").document(user_id).get()
    if not user_snap.exists:
        return jsonify({"error": f"User not found: {user_id}"}), 404

    # ✅ you said users/{user_id}.team_id
    teamId = (user_snap.to_dict() or {}).get("teamId")
    if not teamId:
        return jsonify({"error": f"No team_id found for user {user_id}"}), 400

    price_id = STRIPE_PRICE_ID_DISCOUNT if purchase else STRIPE_PRICE_ID
    if purchase and not STRIPE_PRICE_ID_DISCOUNT:
        return jsonify({"error": "Missing STRIPE_PRICE_ID_DISCOUNT"}), 500

    intended_trial = (not purchase)
    plan = "discount_no_trial" if purchase else "trial"

    # ✅ Only mark attempt/intention — NOT access
    db.collection("teams").document(teamId).set(
        {
            "billing": {
                "access_source": "stripe",
                "state": "checkout_started",
                "status": "pending_payment",          # ✅ consistent canonical state
                "intended_trial": intended_trial,
                "plan": plan,
                "price_id": price_id,
                "checkout_started_by": user_id,
                "checkout_started_at": firestore.SERVER_TIMESTAMP,
                "updated_at": firestore.SERVER_TIMESTAMP,
            }
        },
        merge=True,
    )

    try:
        session = stripe.checkout.Session.create(
            mode="subscription",
            line_items=[{"price": price_id, "quantity": 1}],
            subscription_data={
                **({} if purchase else {"trial_period_days": 7}),
                "metadata": {
                    "teamId": teamId,
                    "user_id": user_id,
                    "plan": plan,
                    "price_id": price_id,
                },
            },
            client_reference_id=teamId,
            metadata={
                "teamId": teamId,
                "user_id": user_id,
                "plan": plan,
                "price_id": price_id,
                "intended_trial": "true" if intended_trial else "false",
            },
            success_url=SUCCESS_URL,
            cancel_url=CANCEL_URL,
        )
    except stripe.error.StripeError as e:
        return jsonify({"error": "Stripe error creating checkout session", "details": str(e)}), 400

    return jsonify({"url": session.url, "team_id": teamId}), 200

# building the response webhook
# Map Stripe subscription statuses into your canonical set


STRIPE_WEBHOOK_SECRET=os.environ.get("STRIPE_WEBHOOK_SECRET")

def now_server_ts():
    return firestore.SERVER_TIMESTAMP

def canonical_status(stripe_status: str) -> str:
    if not stripe_status:
        return "none"
    s = stripe_status.lower()

    if s == "trialing":
        return "trialing"
    if s == "active":
        return "active"
    if s in ("canceled", "unpaid"):
        return "canceled"

    # ✅ IMPORTANT: incomplete is not past_due
    if s in ("incomplete", "incomplete_expired"):
        return "pending_payment"

    if s in ("past_due",):
        return "past_due"

    if s in ("paused",):
        return "paused"

    return "none"

def upsert_team_billing(team_id: str, update: dict):
    update["updated_at"] = now_server_ts()
    db.collection("teams").document(team_id).set({"billing": update}, merge=True)

def write_billing_from_subscription(team_id: str, subscription: dict, state: str):
    subscription_id = subscription.get("id")
    customer_id = subscription.get("customer")
    stripe_status = subscription.get("status")
    status = canonical_status(stripe_status)

    trial_end = subscription.get("trial_end")
    current_period_end = subscription.get("current_period_end")
    cancel_at_period_end = subscription.get("cancel_at_period_end")
    cancel_at = subscription.get("cancel_at")
    canceled_at = subscription.get("canceled_at")
    ended_at = subscription.get("ended_at")

    md = subscription.get("metadata") or {}
    plan = md.get("plan")
    price_id = md.get("price_id")

    # fallback: infer price_id from subscription items if metadata missing
    if not price_id:
        items = ((subscription.get("items") or {}).get("data") or [])
        if items and (items[0].get("price") or {}).get("id"):
            price_id = items[0]["price"]["id"]

    access_expires_at = trial_end if status == "trialing" else current_period_end

    upsert_team_billing(team_id, {
        "access_source": "stripe",
        "state": state,

        "stripe_customer_id": customer_id,
        "stripe_subscription_id": subscription_id,

        # raw + canonical
        "stripe_status": stripe_status,
        "status": status,

        "plan": plan,
        "price_id": price_id,

        "trial_ends_at": trial_end,
        "current_period_end": current_period_end,

        "cancel_at_period_end": bool(cancel_at_period_end),
        "cancel_at": cancel_at,
        "canceled_at": canceled_at,
        "ended_at": ended_at,

        "will_renew": bool(status in ("trialing", "active") and not cancel_at_period_end),
        "access_expires_at": access_expires_at,
    })

def team_id_from_subscription(subscription_obj) -> str | None:
    md = (subscription_obj.get("metadata") or {})
    return md.get("teamId")


def find_team_id_by_subscription_id(subscription_id: str) -> str | None:
    if not subscription_id:
        return None
    matches = (
        db.collection("teams")
          .where("billing.stripe_subscription_id", "==", subscription_id)
          .limit(1)
          .get()
    )
    return matches[0].id if matches else None


def _log(obj, label=""):
    try:
        print(f"\n===== {label} =====")
        print(json.dumps(obj, indent=2, default=str)[:8000])  # cap to avoid huge logs
    except Exception as e:
        print(f"\n===== {label} (failed to json dump: {e}) =====")
        print(str(obj)[:8000])

@app.route("/api/stripe/webhook", methods=["POST"])
def stripe_webhook():
    raw = request.get_data(as_text=False)
    sig_header = request.headers.get("Stripe-Signature", "")

    # Helpful header log (signature only)
    print("\n🔥 STRIPE WEBHOOK HIT")
    print("Stripe-Signature present:", bool(sig_header))
    print("Raw bytes len:", len(raw))

    try:
        event = stripe.Webhook.construct_event(
            payload=raw,
            sig_header=sig_header,
            secret=STRIPE_WEBHOOK_SECRET,
        )
    except ValueError:
        print("❌ Invalid payload")
        return jsonify({"error": "Invalid payload"}), 400
    except stripe.error.SignatureVerificationError:
        print("❌ Invalid signature")
        return jsonify({"error": "Invalid signature"}), 400

    event_type = event.get("type")
    obj = (event.get("data") or {}).get("object") or {}

    print("\n✅ Event type:", event_type)
    print("Object type:", obj.get("object"))
    print("Object id:", obj.get("id"))

    # --- checkout.session.completed ---
    if event_type == "checkout.session.completed":
        team_id = obj.get("client_reference_id") or (obj.get("metadata") or {}).get("teamId")
        print("checkout.session.completed teamId:", team_id)
        _log(obj.get("metadata") or {}, "checkout.session.completed metadata")

        if team_id:
            upsert_team_billing(team_id, {
                "access_source": "stripe",
                "state": "checkout_completed",
                "stripe_customer_id": obj.get("customer"),
                "stripe_subscription_id": obj.get("subscription"),
            })

        return jsonify({"received": True}), 200

    # --- subscription events ---
    if event_type in ("customer.subscription.created", "customer.subscription.updated", "customer.subscription.deleted"):
        sub = obj
        sub_id = sub.get("id")
        stripe_status = sub.get("status")

        md = sub.get("metadata") or {}
        team_id = md.get("teamId")
        if not team_id:
            team_id = find_team_id_by_subscription_id(sub_id)

        print("subscription id:", sub_id)
        print("subscription status:", stripe_status)
        _log(md, "subscription.metadata (raw)")
        print("resolved teamId:", team_id)

        if team_id:
            state = (
                "subscription_created" if event_type == "customer.subscription.created"
                else "subscription_updated" if event_type == "customer.subscription.updated"
                else "subscription_deleted"
            )
            write_billing_from_subscription(team_id, sub, state)

        return jsonify({"received": True}), 200

    # --- invoice success (covers invoice_payment.paid + invoice.paid + invoice.payment_succeeded) ---
    if event_type in ("invoice_payment.paid", "invoice.paid", "invoice.payment_succeeded"):
        subscription_id = None
        invoice_id = None

        if obj.get("object") == "invoice_payment":
            invoice_id = obj.get("invoice")
            print("invoice_payment.paid invoice id:", invoice_id)

            if invoice_id:
                invoice = stripe.Invoice.retrieve(invoice_id)
                subscription_id = invoice.get("subscription")
                print("retrieved invoice subscription id:", subscription_id)

                # lightweight invoice log
                _log({
                    "id": invoice.get("id"),
                    "status": invoice.get("status"),
                    "paid": invoice.get("paid"),
                    "subscription": invoice.get("subscription"),
                    "customer": invoice.get("customer"),
                    "total": invoice.get("total"),
                    "payment_intent": invoice.get("payment_intent"),
                }, "invoice (summary)")
        else:
            # invoice object directly
            invoice_id = obj.get("id")
            subscription_id = obj.get("subscription")
            print("invoice event invoice id:", invoice_id)
            print("invoice event subscription id:", subscription_id)

        if not subscription_id:
            print("⚠️ No subscription id found on invoice success event")
            return jsonify({"received": True}), 200

        # Fetch authoritative subscription
        sub = stripe.Subscription.retrieve(subscription_id)
        sub_md = sub.get("metadata") or {}
        team_id = sub_md.get("teamId") or find_team_id_by_subscription_id(subscription_id)

        print("retrieved subscription id:", sub.get("id"))
        print("retrieved subscription status:", sub.get("status"))
        print("retrieved subscription current_period_end:", sub.get("current_period_end"))
        print("retrieved subscription trial_end:", sub.get("trial_end"))
        _log(sub_md, "subscription.metadata (retrieved)")
        print("resolved teamId:", team_id)

        # Optional: log price id from items
        items = ((sub.get("items") or {}).get("data") or [])
        first_price = (items[0].get("price") or {}).get("id") if items else None
        print("subscription first price id:", first_price)

        if team_id:
            write_billing_from_subscription(team_id, sub, "invoice_payment_succeeded")
        else:
            print("⚠️ Could not resolve teamId for subscription:", subscription_id)

        return jsonify({"received": True}), 200

    # --- invoice payment failed ---
    if event_type in ("invoice.payment_failed", "invoice_payment.failed"):
        subscription_id = None
        invoice_id = None

        if obj.get("object") == "invoice_payment":
            invoice_id = obj.get("invoice")
            print("invoice_payment.failed invoice id:", invoice_id)
            if invoice_id:
                invoice = stripe.Invoice.retrieve(invoice_id)
                subscription_id = invoice.get("subscription")
                print("retrieved invoice subscription id:", subscription_id)
        else:
            invoice_id = obj.get("id")
            subscription_id = obj.get("subscription")

        if subscription_id:
            team_id = find_team_id_by_subscription_id(subscription_id)
            print("resolved teamId:", team_id)

            if team_id:
                upsert_team_billing(team_id, {
                    "access_source": "stripe",
                    "state": "invoice_payment_failed",
                    "status": "past_due",
                    "will_renew": False,
                })

        return jsonify({"received": True}), 200

    print("ℹ️ Ignored event type:", event_type)
    return jsonify({"received": True, "ignored": event_type}), 200

@app.route("/api/stripe/create_portal_session", methods=["POST"])
def create_portal_session():
    PORTAL_RETURN_URL = os.environ.get("PORTAL_RETURN_URL")
    """
    Body:
      { "user_id": "<firebase_uid>" }

    Returns:
      { "url": "<stripe_customer_portal_url>" }
    """
    data = request.get_json(silent=True) or {}
    user_id = data.get("user_id")

    if not user_id:
        return jsonify({"error": "Missing user_id"}), 400

    # 1) Look up user -> team_id
    user_snap = db.collection("users").document(user_id).get()
    if not user_snap.exists:
        return jsonify({"error": f"User not found: {user_id}"}), 404

    team_id = (user_snap.to_dict() or {}).get("teamId")
    if not team_id:
        return jsonify({"error": f"No team_id found for user {user_id}"}), 400

    # 2) Look up team's stripe_customer_id
    team_snap = db.collection("teams").document(team_id).get()
    if not team_snap.exists:
        return jsonify({"error": f"Team not found: {team_id}"}), 404

    billing = (team_snap.to_dict() or {}).get("billing") or {}
    customer_id = billing.get("stripe_customer_id")

    if not customer_id:
        return jsonify({"error": "No stripe_customer_id found for this team"}), 400

    # 3) Create Stripe Customer Portal session
    try:
        session = stripe.billing_portal.Session.create(
            customer=customer_id,
            return_url=PORTAL_RETURN_URL,
        )
    except stripe.error.StripeError as e:
        return jsonify(
            {
                "error": "Failed to create Stripe portal session",
                "details": str(e),
            }
        ), 400

    return jsonify({"url": session.url}), 200

@app.route("/settings/manage_builling", methods=["POST"])
@require_solari_key
def manage_builling():
    """
    Body:
      { "user_id": "<firebase_uid>" }

    Behavior:
      - If billing.access_source == "calendar", start Stripe Checkout.
      - If billing.access_source == "stripe", open Stripe Customer Portal.
    """
    data = request.get_json(silent=True) or {}
    user_id = data.get("user_id")

    if not user_id:
        return jsonify({"error": "Missing user_id"}), 400

    user_snap = db.collection("users").document(user_id).get()
    if not user_snap.exists:
        return jsonify({"error": f"User not found: {user_id}"}), 404

    team_id = (user_snap.to_dict() or {}).get("teamId")
    if not team_id:
        return jsonify({"error": f"No team_id found for user {user_id}"}), 400

    team_snap = db.collection("teams").document(team_id).get()
    if not team_snap.exists:
        return jsonify({"error": f"Team not found: {team_id}"}), 404

    billing = (team_snap.to_dict() or {}).get("billing") or {}
    access_source = (billing.get("access_source") or "").strip().lower()
    billing_status = (billing.get("status") or "").strip().lower()

    if access_source == "calendar" or (access_source == "stripe" and billing_status == "pending_payment"):
        price_id = STRIPE_PRICE_ID
        intended_trial = True
        plan = "trial"

        db.collection("teams").document(team_id).set(
            {
                "billing": {
                    "state": "checkout_started",
                    "status": "pending_payment",
                    "intended_trial": intended_trial,
                    "plan": plan,
                    "price_id": price_id,
                    "checkout_started_by": user_id,
                    "checkout_started_at": firestore.SERVER_TIMESTAMP,
                    "updated_at": firestore.SERVER_TIMESTAMP,
                }
            },
            merge=True,
        )

        try:
            session = stripe.checkout.Session.create(
                mode="subscription",
                line_items=[{"price": price_id, "quantity": 1}],
                subscription_data={
                    "trial_period_days": 7,
                    "metadata": {
                        "teamId": team_id,
                        "user_id": user_id,
                        "plan": plan,
                        "price_id": price_id,
                    },
                },
                client_reference_id=team_id,
                metadata={
                    "teamId": team_id,
                    "user_id": user_id,
                    "plan": plan,
                    "price_id": price_id,
                    "intended_trial": "true",
                },
                success_url=SUCCESS_URL,
                cancel_url=CANCEL_URL,
            )
        except stripe.error.StripeError as e:
            return jsonify({"error": "Stripe error creating checkout session", "details": str(e)}), 400

        return jsonify({"url": session.url, "team_id": team_id, "flow": "stripe_checkout"}), 200

    if access_source == "stripe":
        PORTAL_RETURN_URL = os.environ.get("PORTAL_RETURN_URL")
        customer_id = billing.get("stripe_customer_id")
        if not customer_id:
            return jsonify({"error": "No stripe_customer_id found for this team"}), 400
        try:
            session = stripe.billing_portal.Session.create(
                customer=customer_id,
                return_url=PORTAL_RETURN_URL,
            )
        except stripe.error.StripeError as e:
            return jsonify({"error": "Failed to create Stripe portal session", "details": str(e)}), 400

        return jsonify({"url": session.url, "team_id": team_id, "flow": "stripe_portal"}), 200

    return jsonify({"error": "Unsupported billing access_source", "access_source": access_source}), 400

# Booking - calendar route code
CAL_WEBHOOK_SECRET = os.environ.get("SOLARI_INTERNAL_KEY")  # same internal key you set in Cal
CAL_EVENT_URL = os.environ.get(
    "CAL_EVENT_URL",
    "https://cal.com/sahil-sinha-hugr4z/solari-onboarding"
)

@app.route("/api/cal/create_appointment", methods=["POST"])
def cal_create_appointment():
    """
    Body:
      { "user_id": "<firebase_uid>" }

    Behavior:
      - users/{user_id}.team_id -> team_id
      - teams/{team_id}.billing.status = "pending_booking"
      - returns cal_url with metadata[team_id], metadata[user_id]
    """
    data = request.get_json(silent=True) or {}
    user_id = data.get("user_id")
    if not user_id:
        return jsonify({"error": "Missing user_id"}), 400

    # Look up user -> team_id
    user_snap = db.collection("users").document(user_id).get()
    if not user_snap.exists:
        return jsonify({"error": f"User not found: {user_id}"}), 404

    team_id = (user_snap.to_dict() or {}).get("teamId")
    if not team_id:
        return jsonify({"error": f"No team_id found for user {user_id}"}), 400

    # Mark pending booking on team
    db.collection("teams").document(team_id).set(
        {
            "billing": {
                "access_source": "calendar",
                "status": "pending_booking",
                "pending_booking_user_id": user_id,
                "updated_at": firestore.SERVER_TIMESTAMP,
            }
        },
        merge=True,
    )

    # Build Cal URL with metadata
    params = {
        "metadata[team_id]": team_id,
        "metadata[user_id]": user_id,
    }
    cal_url = CAL_EVENT_URL + "?" + urllib.parse.urlencode(params)

    return jsonify({"cal_url": cal_url, "team_id": team_id}), 200

@app.route("/api/webhooks/cal", methods=["POST"])
def cal_webhook():
    payload = request.get_json(silent=True) or {}

    print("\n====== CAL WEBHOOK PAYLOAD ======")
    print(json.dumps(payload, indent=2, default=str))

    trigger = payload.get("triggerEvent")
    p = payload.get("payload") or {}
    metadata = p.get("metadata") or {}

    # Ignore pings/tests
    if trigger in (None, "PING"):
        return jsonify({"received": True, "ignored": trigger}), 200

    team_id = metadata.get("team_id")
    user_id = metadata.get("user_id")

    if not team_id or not user_id:
        # Helpful logging for debugging unexpected payloads
        print("❌ Cal webhook missing team_id/user_id in payload.metadata")
        print(json.dumps(payload, indent=2))
        return jsonify({"error": "Missing team_id/user_id in payload.metadata"}), 400

    booking_id = p.get("bookingId")
    booking_status = p.get("status")  # e.g. "ACCEPTED"
    start_time = p.get("startTime")
    end_time = p.get("endTime")
    uid = payload["payload"]["uid"]

    video_url = metadata.get("videoCallUrl") or (p.get("videoCallData") or {}).get("url")

    # ---- Booking created => grant access
    if trigger == "BOOKING_CREATED":
        db.collection("teams").document(team_id).set(
            {
                "billing": {
                    "access_source": "calendar",
                    "status": "booking_confirmed",
                    "booked_by_user_id": user_id,

                    "cal_booking_id": booking_id,
                    "reschedule_id": uid,
                    "cal_booking_status": booking_status,
                    "cal_start_time": start_time,
                    "cal_end_time": end_time,
                    "cal_video_url": video_url,

                    "updated_at": firestore.SERVER_TIMESTAMP,
                }
            },
            merge=True,
        )
        return jsonify({"received": True}), 200

    # ---- Booking cancelled => revoke access until they book again
    if trigger == "BOOKING_CANCELLED":
        db.collection("teams").document(team_id).set(
            {
                "billing": {
                    "access_source": "calendar",
                    "status": "booking_cancelled",
                    "booked_by_user_id": user_id,

                    "cal_booking_id": booking_id,
                    "reschedule_id": uid,
                    "cal_booking_status": booking_status,
                    "cal_start_time": start_time,
                    "cal_end_time": end_time,

                    "updated_at": firestore.SERVER_TIMESTAMP,
                }
            },
            merge=True,
        )
        return jsonify({"received": True}), 200

    # Optional: treat reschedule as confirmed (or ignore)
    if trigger == "BOOKING_RESCHEDULED":
        db.collection("teams").document(team_id).set(
            {
                "billing": {
                    "access_source": "calendar",
                    "status": "booking_confirmed",
                    "booked_by_user_id": user_id,

                    "cal_booking_id": booking_id,
                    "reschedule_id": uid,
                    "cal_booking_status": booking_status,
                    "cal_start_time": start_time,
                    "cal_end_time": end_time,
                    "cal_video_url": video_url,

                    "updated_at": firestore.SERVER_TIMESTAMP,
                }
            },
            merge=True,
        )
        return jsonify({"received": True}), 200

    # Ignore other events
    return jsonify({"received": True, "ignored": trigger}), 200


@app.route("/auth/jira/callback", methods=['GET'])
def jira_oauth_callback():
    """
    OAuth callback from Atlassian after user approves access.
    This endpoint is called by Atlassian after user authorizes the app.
    """
    # 1. Get ?code= from Atlassian redirect
    code = request.args.get("code")
    uid = request.args.get("state")  # ✅ state contains user_id from connect step

    logger.info(f"Callback received - code: {bool(code)}, uid: {uid}")

    if not code:
        return jsonify({"error": "Missing ?code in callback URL"}), 400
    if not uid:
        return jsonify({"error": "Missing uid in OAuth state parameter"}), 400

    # For testing: Get client credentials from environment variables
    # TODO: In production, get these from Firebase using the uid
    client_id = os.getenv('JIRA_CLIENT_ID')
    client_secret = os.getenv('JIRA_CLIENT_SECRET')
    
    if not client_id or not client_secret:
        return jsonify({"error": "Jira client credentials not configured"}), 500

    # 2. Exchange code for access + refresh tokens
    token_url = "https://auth.atlassian.com/oauth/token"
    token_payload = {
        "grant_type": "authorization_code",
        "client_id": client_id,
        "client_secret": client_secret,
        "code": code,
        "redirect_uri": ATLASSIAN_REDIRECT_URI,
    }

    try:
        token_res = requests.post(token_url, json=token_payload)
        if token_res.status_code != 200:
            logger.error(f"Token exchange failed: {token_res.text}")
            return jsonify({"error": "Failed to exchange code for tokens", "details": token_res.json()}), 400

        token_data = token_res.json()
        access_token = token_data.get("access_token")
        refresh_token = token_data.get("refresh_token")
        expires_in = token_data.get("expires_in", 3600)  # ≈1 hour

        logger.info(f"Got tokens - access_token: {bool(access_token)}, refresh_token: {bool(refresh_token)}")
        logger.info(f"Access token: {access_token}")
        logger.info(f"Refresh token: {refresh_token}")

        # 3. Get user's Jira Cloud ID
        resources_url = "https://api.atlassian.com/oauth/token/accessible-resources"
        resources_res = requests.get(
            resources_url,
            headers={"Authorization": f"Bearer {access_token}"}
        )

        if resources_res.status_code != 200:
            logger.error(f"Failed to fetch resources: {resources_res.text}")
            return jsonify({"error": "Failed to fetch Jira cloud ID", "details": resources_res.json()}), 400

        resources = resources_res.json()
        if not resources:
            return jsonify({"error": "No Jira site accessible with this account"}), 400

        jira_site = resources[0]
        cloud_id = jira_site.get("id")
        jira_site_url = jira_site.get("url")

        logger.info(f"Got Jira info - cloud_id: {cloud_id}, site_url: {jira_site_url}")

        # 4. ✅ Save tokens securely to Firestore (team-scoped)
        try:
            db = firestore.client()
            team_id = get_team_id_for_uid(db, uid)
            db.collection("teams").document(team_id).set({
                "jira_access_token": access_token,
                "jira_refresh_token": refresh_token,
                "jira_cloud_id": cloud_id,
                "jira_site_url": jira_site_url,
                "jira_connected": True,
                "jira_connected_at": firestore.SERVER_TIMESTAMP,
                "jira_expires_at": datetime.utcnow() + timedelta(seconds=expires_in)
            }, merge=True)
            logger.info(f"✅ Jira OAuth saved in Firestore for team {team_id} (user {uid})")
        except KeyError as e:
            logger.error(f"Failed to resolve team for user {uid}: {str(e)}")
            return jsonify({"error": "Failed to resolve team for user"}), 400
        except Exception as e:
            logger.error(f"Failed to save OAuth tokens for user {uid}: {str(e)}")
            return jsonify({"error": "Failed to save OAuth credentials"}), 500

        # Return success response (without tokens for security)
        # return jsonify({
        #     "success": True,
        #     "message": "Jira OAuth successful",
        #     "uid": uid,
        #     "cloud_id": cloud_id,
        #     "jira_site_url": jira_site_url
        # }), 200

        # 5. Redirect user back to frontend with success message
        return redirect(f"{FRONTEND_SUCCESS_URL}?status=jira_connected")

    except requests.exceptions.RequestException as e:
        logger.error(f"Request error during OAuth callback: {str(e)}")
        return jsonify({"error": f"OAuth request failed: {str(e)}"}), 500
    except Exception as e:
        logger.error(f"Unexpected error during OAuth callback: {str(e)}")
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

# ============================================================================
# JIRA TICKET QUERY CODE (using OAuth credentials)
# ============================================================================

def _refresh_jira_access_token(user_id: str, refresh_token: str) -> dict:
    """
    Refresh Jira access token using refresh token.
    
    Args:
        user_id: Firebase user ID
        refresh_token: Jira refresh token
    
    Returns:
        dict with new access_token, refresh_token, expires_in
    
    Raises:
        Exception if refresh fails
    """
    client_id = os.getenv('JIRA_CLIENT_ID')
    client_secret = os.getenv('JIRA_CLIENT_SECRET')
    
    if not client_id or not client_secret:
        raise Exception("Jira client credentials not configured")
    
    token_url = "https://auth.atlassian.com/oauth/token"
    token_payload = {
        "grant_type": "refresh_token",
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token
    }
    
    response = requests.post(token_url, json=token_payload)
    
    if response.status_code != 200:
        error_msg = f"Failed to refresh access token: {response.text}"
        logger.error(error_msg)
        raise Exception(error_msg)
    
    token_data = response.json()
    return {
        "access_token": token_data.get("access_token"),
        "refresh_token": token_data.get("refresh_token", refresh_token),  # May not be returned
        "expires_in": token_data.get("expires_in", 3600)
    }

def get_atlassian_access_token(user_id: str) -> str:
    """
    Get a valid Atlassian access token for a team, refreshing if expired.
    
    Args:
        user_id: Firebase user ID (used to resolve team)
    
    Returns:
        Valid access token string
    
    Raises:
        Exception if user/team not found, credentials missing, or refresh fails
    """
    db = firestore.client()
    try:
        team_id = get_team_id_for_uid(db, user_id)
    except KeyError:
        raise Exception(f"User {user_id} not found in Firestore")

    doc_ref = db.collection("teams").document(team_id)
    doc = doc_ref.get()
    
    if not doc.exists:
        raise Exception(f"Team {team_id} not found in Firestore")
    
    user_data = doc.to_dict() or {}
    access_token = user_data.get("jira_access_token")
    refresh_token = user_data.get("jira_refresh_token")
    expires_at = user_data.get("jira_expires_at")
    
    if not access_token or not refresh_token:
        raise Exception(f"Jira credentials not found for team {team_id}. Please complete OAuth flow.")
    
    # Check if token is expired
    # expires_at is a Firestore timestamp, convert to datetime if needed
    if expires_at:
        if hasattr(expires_at, 'timestamp'):  # Firestore Timestamp
            expires_datetime = datetime.fromtimestamp(expires_at.timestamp())
        elif isinstance(expires_at, datetime):
            expires_datetime = expires_at
        else:
            # Assume it's already a datetime
            expires_datetime = expires_at
        
        # Add 60 second buffer to refresh before actual expiration
        if datetime.utcnow() >= (expires_datetime - timedelta(seconds=60)):
            logger.info(f"Access token expired for team {team_id}, refreshing...")
            
            # Refresh the token
            new_tokens = _refresh_jira_access_token(user_id, refresh_token)
            new_access_token = new_tokens["access_token"]
            new_refresh_token = new_tokens.get("refresh_token", refresh_token)
            expires_in = new_tokens.get("expires_in", 3600)
            
            # Update Firestore with new tokens
            doc_ref.update({
                "jira_access_token": new_access_token,
                "jira_refresh_token": new_refresh_token,
                "jira_expires_at": datetime.utcnow() + timedelta(seconds=expires_in)
            })
            
            logger.info(f"Successfully refreshed access token for team {team_id}")
            return new_access_token
    
    # Token is still valid
    return access_token

def _get_jira_creds(user_id: str):
    """
    Retrieve Jira OAuth credentials from Firestore for a team.
    Automatically refreshes token if expired.
    
    Returns:
        dict with keys: access_token, cloud_id, site_url
    Raises:
        Exception if user/team not found or credentials missing
    """
    db = firestore.client()
    try:
        team_id = get_team_id_for_uid(db, user_id)
    except KeyError:
        raise Exception(f"User {user_id} not found in Firestore")

    user_doc = db.collection("teams").document(team_id).get()
    
    if not user_doc.exists:
        raise Exception(f"Team {team_id} not found in Firestore")
    
    user_data = user_doc.to_dict() or {}
    cloud_id = user_data.get("jira_cloud_id")
    
    if not cloud_id:
        raise Exception(f"Jira credentials not found for team {team_id}. Please complete OAuth flow.")
    
    # Get valid access token (refreshes if needed)
    access_token = get_atlassian_access_token(user_id)
    
    return {
        "access_token": access_token,
        "cloud_id": cloud_id,
        "site_url": user_data.get("jira_site_url")
    }

def _merge_jira_tickets(existing_tickets: list, new_tickets: list) -> list:
    """
    Merge Jira tickets by key/id, preferring new ticket data.
    Preserves existing order, then appends new tickets.
    """
    merged = {}
    for ticket in existing_tickets or []:
        if not isinstance(ticket, dict):
            continue
        ticket_key = ticket.get("key") or ticket.get("id")
        if ticket_key:
            merged[ticket_key] = ticket
    for ticket in new_tickets or []:
        if not isinstance(ticket, dict):
            continue
        ticket_key = ticket.get("key") or ticket.get("id")
        if ticket_key:
            merged[ticket_key] = ticket
    return list(merged.values())

def _build_jira_ticket_text(ticket: dict) -> str:
    ticket_id = ticket.get("id")
    assignee = ticket.get("assignee") or {}
    issuetype = ticket.get("issuetype")
    ticket_key = ticket.get("key")
    project = ticket.get("project") or {}
    status = ticket.get("status")
    summary = ticket.get("summary")

    assignee_name = assignee.get("displayName") if isinstance(assignee, dict) else None
    project_key = project.get("key") if isinstance(project, dict) else None
    project_name = project.get("name") if isinstance(project, dict) else None

    lines = [
        f"jira ticket id: {ticket_id or ''}".strip(),
        f"assigned: {assignee_name or ''}".strip(),
        f"issuetype: {issuetype or ''}".strip(),
        f"key: {ticket_key or ''}".strip(),
        f"project key: {project_key or ''}".strip(),
        f"project name: {project_name or ''}".strip(),
        f"status: {status or ''}".strip(),
        f"summary: {summary or ''}".strip(),
    ]

    return "\n".join(lines).strip()

def _remove_jira_ticket_by_id(existing_tickets: list, jira_id: str) -> list:
    if not existing_tickets:
        return []
    return [
        ticket for ticket in existing_tickets
        if isinstance(ticket, dict) and str(ticket.get("id")) != str(jira_id)
    ]

def _get_team_pinecone_namespace(db, team_id: str) -> str:
    team_doc = db.collection("teams").document(team_id).get()
    if not team_doc.exists:
        raise KeyError("team_not_found")
    namespace = (team_doc.to_dict() or {}).get("pinecone_namespace")
    if not namespace:
        raise KeyError("pinecone_namespace_not_found")
    return namespace

def _delete_firestore_document_recursive(doc_ref) -> None:
    """
    Recursively delete a Firestore document and all nested subcollections.
    """
    for subcol in doc_ref.collections():
        for subdoc in subcol.stream():
            _delete_firestore_document_recursive(subdoc.reference)
    doc_ref.delete()

def _fetch_confluence_page_storage_body(user_id: str, page_id: str, base_url: str, solari_key: Optional[str]) -> str:
    """
    Fetch Confluence page body in storage format by calling the internal /api/confluence/get-page endpoint.
    Returns the body.storage.value string (or empty string if missing).
    """
    url = f"{base_url}/api/confluence/get-page"
    headers = {"Accept": "application/json"}
    if solari_key:
        headers["x-solari-key"] = solari_key
    params = {
        "user_id": user_id,
        "page_id": page_id,
        "body_format": "storage",
    }

    response = requests.get(url, headers=headers, params=params, timeout=45)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch Confluence page body: {response.text}")
    page_data = (response.json() or {}).get("page") or {}
    body = (page_data.get("body") or {}).get("storage") or {}
    return body.get("value") or ""

def _jira_request(user_id: str, method: str, endpoint: str, params: dict = None, json_body: dict = None):
    """
    Make an authenticated request to Jira API.
    
    Args:
        user_id: Firebase user ID
        method: HTTP method (GET, POST, etc.)
        endpoint: API endpoint path (e.g., "/rest/api/3/issue/picker")
        params: Query parameters (for GET requests)
        json_body: JSON body (for POST requests)
    
    Returns:
        Response JSON data
    Raises:
        Exception on request failure
    """
    creds = _get_jira_creds(user_id)
    access_token = creds["access_token"]
    cloud_id = creds["cloud_id"]
    
    # Construct Jira API URL: https://api.atlassian.com/ex/jira/{cloud_id}/rest/api/3/...
    base_url = f"https://api.atlassian.com/ex/jira/{cloud_id}"
    url = f"{base_url}{endpoint}"
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json"
    }
    
    if json_body:
        headers["Content-Type"] = "application/json"
    
    response = requests.request(
        method=method,
        url=url,
        headers=headers,
        params=params,
        json=json_body
    )
    
    if response.status_code >= 400:
        error_msg = f"Jira API error ({response.status_code}): {response.text}"
        logger.error(error_msg)
        raise Exception(error_msg)
    
    return response.json()

def _search_by_picker(user_id: str, query_text: str):
    """
    Simple search using the Issue Picker endpoint.
    Returns list of numeric IDs as strings.
    Good for plain text searches like "PROJ-123" or "bug in login"
    """
    data = _jira_request(
        user_id,
        "GET",
        "/rest/api/3/issue/picker",
        params={"query": query_text},
    )
    ids = []
    for section in data.get("sections", []):
        for issue in section.get("issues", []):
            # picker returns numeric ID, ensure str for consistency
            if "id" in issue:
                ids.append(str(issue["id"]))
    return ids

def _search_by_jql(user_id: str, jql: str, max_results: int = 20):
    """
    JQL search via the new endpoint. Returns list of string IDs.
    Note: We're using POST /search/jql (required by latest API).
    
    Examples:
    - "project = PROJ"
    - "project = PROJ AND status = 'In Progress'"
    - "assignee = currentUser() AND created >= -7d"
    """
    body = {"jql": jql, "maxResults": max_results}
    data = _jira_request(user_id, "POST", "/rest/api/3/search/jql", json_body=body)
    issues = data.get("issues", [])
    return [i.get("id") for i in issues if i.get("id")]

def _get_issue(user_id: str, issue_id_or_key: str):
    """
    Fetch full details for a specific issue by ID or key.
    Examples: "12345" (ID) or "PROJ-123" (key)
    """
    return _jira_request(user_id, "GET", f"/rest/api/3/issue/{issue_id_or_key}")

def _filter_ticket_fields(ticket_data):
    """
    Filter ticket data to keep only specific fields.
    Accepts list of issues or a single issue dict.
    """
    fields_to_keep = [
        "assignee", "attachment", "comment", "created", "creator",
        "duedate", "issuetype", "priority", "project", "status",
        "summary", "subtasks",
    ]
    top_level_fields = ["id", "key", "self"]
    
    def _filter_one(ticket: dict):
        out = {}
        for k in top_level_fields:
            if k in ticket:
                out[k] = ticket[k]
        if "fields" in ticket:
            out["fields"] = {}
            for k in fields_to_keep:
                if k in ticket["fields"]:
                    out["fields"][k] = ticket["fields"][k]
        return out
    
    if isinstance(ticket_data, list):
        return [_filter_one(t) for t in ticket_data]
    return _filter_one(ticket_data)

def get_filtered_full_ticket_info_oauth(search_input: str, search_type: str, user_id: str):
    """
    Main function to query Jira tickets using OAuth credentials.
    
    Args:
        search_input: JQL query or plain text search string
        search_type: 'jql' or 'query'
        user_id: Firebase user ID (used to retrieve OAuth credentials)
    
    Returns:
        {
            "success": True/False,
            "data": [list of filtered ticket objects],
            "ticket_count": int,
            "query_type": "JQL" or "Search Query",
            "error": str (if success=False)
        }
    
    Examples:
        # JQL search
        result = get_filtered_full_ticket_info_oauth(
            "project = PROJ AND status = 'In Progress'",
            "jql",
            "test_user_123"
        )
        
        # Plain text search
        result = get_filtered_full_ticket_info_oauth(
            "PROJ-123",
            "query",
            "test_user_123"
        )
    """
    try:
        if not search_input:
            return {"success": False, "error": "search_input is required"}
        search_type_norm = (search_type or "").strip().lower()
        if search_type_norm not in ("jql", "query"):
            return {"success": False, "error": "search_type must be 'jql' or 'query'"}
        
        # Step 1: Search for ticket IDs
        if search_type_norm == "jql":
            ticket_ids = _search_by_jql(user_id, search_input)
            query_label = "JQL"
        else:
            ticket_ids = _search_by_picker(user_id, search_input)
            query_label = "Search Query"
        
        # Step 2: Fetch full issue details for each ID
        details = []
        for issue_id in ticket_ids:
            try:
                details.append(_get_issue(user_id, issue_id))
            except Exception as e:
                # continue on per-issue failure
                logger.warning(f"Failed fetching issue {issue_id}: {e}")
        
        # Step 3: Filter fields and return
        filtered = _filter_ticket_fields(details)
        return {
            "success": True,
            "data": filtered,
            "ticket_count": len(filtered),
            "query_type": query_label,
        }
    
    except Exception as e:
        return {"success": False, "error": f"Request failed: {str(e)}"}

@app.route("/api/jira/search", methods=["POST"])
@require_solari_key
def jira_search():
    """
    HTTP endpoint to search Jira tickets.
    
    Request Body:
    {
      "user_id": "firebase_uid",
      "search_type": "jql" | "query",
      "search_input": "<JQL or plain text>"
    }
    
    Response:
    {
      "status": "success" | "failure",
      "data": [ticket objects],
      "ticket_count": int,
      "query_type": "JQL" | "Search Query",
      "error": str (if failure)
    }
    """
    try:
        body = request.get_json(force=True) or {}
        user_id = body.get("user_id")
        search_input = body.get("search_input")
        search_type = body.get("search_type")  # "jql" or "query"
        
        if not user_id:
            return jsonify({"status": "failure", "error": "user_id is required"}), 400
        
        if not search_input:
            return jsonify({"status": "failure", "error": "search_input is required"}), 400
        
        if (search_type or "").lower() not in ("jql", "query"):
            return jsonify({"status": "failure", "error": "search_type must be 'jql' or 'query'"}), 400
        
        result = get_filtered_full_ticket_info_oauth(search_input, search_type, user_id)
        
        if result.get("success"):
            return jsonify({
                "status": "success",
                "data": result["data"],
                "ticket_count": result["ticket_count"],
                "query_type": result["query_type"],
            }), 200
        else:
            return jsonify({"status": "failure", "error": result.get("error", "Unknown error")}), 500
    
    except Exception as e:
        logger.error(f"Jira search error: {str(e)}")
        return jsonify({"status": "failure", "error": f"Internal server error: {str(e)}"}), 500

@app.route("/api/jira/workspaces", methods=["GET"])
@require_solari_key
def jira_get_workspaces():
    """
    Returns a list of Jira workspaces (cloud sites) that the authenticated user has access to.
    Requires: user_id query parameter
    
    Usage:
        GET /api/jira/workspaces?user_id=test_user_123
    
    Response:
        {
            "status": "success",
            "workspaces": [
                {
                    "name": "My Jira Site",
                    "url": "https://solariai.atlassian.net",
                    "cloudId": "7ad8f09d-6dc0-444a-acf1-f55b3666dbb6",
                    "scopes": ["read:jira-work", "write:jira-work"]
                },
                ...
            ]
        }
    """
    user_id = request.args.get("user_id")
    
    if not user_id:
        return jsonify({"status": "failure", "error": "user_id is required"}), 400
    
    try:
        # --- Get credentials from Firestore ---
        creds = _get_jira_creds(user_id)
        access_token = creds["access_token"]
        
        # --- Call Jira to get accessible resources ---
        url = "https://api.atlassian.com/oauth/token/accessible-resources"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json"
        }
        
        response = requests.get(url, headers=headers)
        
        # TODO: If token expired, implement token refresh logic here
        # if response.status_code == 401:
        #     new_access = _refresh_access_token(user_id, creds["refresh_token"])
        #     headers["Authorization"] = f"Bearer {new_access}"
        #     response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            logger.error(f"Failed to fetch Jira workspaces: {response.text}")
            return jsonify({
                "status": "failure",
                "error": f"Failed to fetch Jira workspaces: {response.text}"
            }), 400
        
        workspaces = response.json()
        
        # Filter to only Jira sites (not Confluence, etc.)
        jira_sites = [
            {
                "name": w.get("name"),
                "url": w.get("url"),
                "cloudId": w.get("id"),
                "scopes": w.get("scopes", [])
            }
            for w in workspaces if "jira" in ",".join(w.get("scopes", [])).lower()
        ]
        
        return jsonify({
            "status": "success",
            "workspaces": jira_sites
        }), 200
    
    except Exception as e:
        logger.error(f"Jira workspaces error: {str(e)}")
        return jsonify({
            "status": "failure",
            "error": str(e)
        }), 500

@app.route("/api/jira/add_ticket", methods=["POST"])
@require_solari_key
def jira_add_ticket():
    """
    Enqueue Jira ingestion job (v1.5).
    Request Body:
    {
      "user_id": "firebase_uid",
      "agent_id": "agent_id",
      "tickets": [ { ...ticket fields... } ]
    }
    """
    body = request.get_json(force=True) or {}
    user_id = body.get("user_id")
    agent_id = body.get("agent_id")
    tickets = body.get("tickets")

    if not user_id:
        return jsonify({"status": "failure", "error": "user_id is required"}), 400
    if not agent_id:
        return jsonify({"status": "failure", "error": "agent_id is required"}), 400
    if not isinstance(tickets, list) or not tickets:
        return jsonify({"status": "failure", "error": "tickets must be a non-empty list"}), 400

    for t in tickets:
        if not isinstance(t, dict):
            return jsonify({"status": "failure", "error": "each ticket must be an object"}), 400
        if not t.get("id"):
            return jsonify({"status": "failure", "error": "each ticket must include id"}), 400

    db = firestore.client()
    team_id = get_team_id_for_uid(db, user_id)

    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(days=30)

    sources = []
    for t in tickets:
        ticket_id = str(t.get("id"))
        fields = t.get("fields") or {}
        project = fields.get("project") or {}
        status = fields.get("status") or {}
        issuetype = fields.get("issuetype") or {}
        assignee = fields.get("assignee") or None

        assignee_payload = None
        if assignee:
            assignee_payload = {
                "displayName": assignee.get("displayName"),
                "accountId": assignee.get("accountId"),
            }

        ticket_payload = {
            "key": t.get("key"),
            "id": t.get("id"),
            "summary": fields.get("summary") or t.get("summary"),
            "project": {
                "key": project.get("key"),
                "name": project.get("name"),
            },
            "status": status.get("name"),
            "issuetype": issuetype.get("name"),
            "assignee": assignee_payload,
            "created": fields.get("created"),
            "self": t.get("self"),
        }
        sources.append({
            "source_key": f"jira:{ticket_id}",
            "type": "jira",
            "id": ticket_id,
            "title": ticket_payload.get("key") or ticket_payload.get("summary") or f"Jira {ticket_id}",
            "url": t.get("url"),
            "ticket": ticket_payload,
            "status": "queued",
            "stage": "queued",
            "checkpoint": {
                "embedded": False,
            },
            "error": None,
            "updated_at": now,  # IMPORTANT: datetime, not SERVER_TIMESTAMP
        })

    job_id = uuid.uuid4().hex
    job_ref = (
        db.collection("teams").document(team_id)
          .collection("upload_jobs").document(job_id)
    )

    job_ref.set({
        "job_type": "ingest_sources",
        "connector": "jira",
        "status": "queued",
        "created_at": firestore.SERVER_TIMESTAMP,
        "updated_at": firestore.SERVER_TIMESTAMP,
        "expires_at": expires_at,

        "locked_by": None,
        "locked_until": None,
        "progress": 0,
        "message": "Queued",
        "created_by_user_id": user_id,

        "team_id": team_id,
        "agent_id": agent_id,

        "sources": sources,
    })

    return jsonify({
        "status": "success",
        "job_id": job_id,
        "team_id": team_id,
        "queued_sources": len(sources),
    }), 200

# @app.route("/api/jira/add_ticket", methods=["POST"])
# @require_solari_key
# def jira_add_ticket():
#     """
#     Add Jira tickets as a source for both agent and team scopes.
    
#     Request Body:
#     {
#       "user_id": "firebase_uid",
#       "agent_id": "agent_id",
#       "tickets": [ { ...ticket fields... } ]
#     }
#     """
#     try:
#         body = request.get_json(force=True) or {}
#         user_id = body.get("user_id")
#         agent_id = body.get("agent_id")
#         tickets = body.get("tickets")
        
#         if not user_id:
#             return jsonify({"status": "failure", "error": "user_id is required"}), 400
#         if not agent_id:
#             return jsonify({"status": "failure", "error": "agent_id is required"}), 400
#         if not isinstance(tickets, list) or not tickets:
#             return jsonify({"status": "failure", "error": "tickets must be a non-empty list"}), 400
        
#         for ticket in tickets:
#             if not isinstance(ticket, dict):
#                 return jsonify({"status": "failure", "error": "each ticket must be an object"}), 400
#             if not ticket.get("id"):
#                 return jsonify({"status": "failure", "error": "each ticket must include id"}), 400
        
#         db = firestore.client()
#         team_id = get_team_id_for_uid(db, user_id)
#         namespace = _get_team_pinecone_namespace(db, team_id)
        
#         # --- Agent-level Jira source ---
#         agent_sources_ref = (
#             db.collection("teams").document(team_id)
#               .collection("agents").document(agent_id)
#               .collection("sources")
#         )
#         agent_jira_docs = agent_sources_ref.where("type", "==", "jira").stream()
#         agent_jira_doc = next(agent_jira_docs, None)
        
#         if agent_jira_doc:
#             existing_agent_tickets = (agent_jira_doc.to_dict() or {}).get("tickets", [])
#             merged_tickets = _merge_jira_tickets(existing_agent_tickets, tickets)
#             agent_sources_ref.document(agent_jira_doc.id).update({
#                 "nickname": "jira",
#                 "tickets": merged_tickets,
#                 "updated_at": firestore.SERVER_TIMESTAMP
#             })
#         else:
#             agent_sources_ref.add({
#                 "type": "jira",
#                 "nickname": "jira",
#                 "agent_id": agent_id,
#                 "tickets": tickets,
#                 "created_at": firestore.SERVER_TIMESTAMP,
#                 "updated_at": firestore.SERVER_TIMESTAMP
#             })
        
#         # --- Team-level Jira source ---
#         team_sources_ref = db.collection("teams").document(team_id).collection("sources")
#         team_jira_docs = team_sources_ref.where("type", "==", "jira").stream()
#         team_jira_doc = next(team_jira_docs, None)
        
#         if team_jira_doc:
#             existing_team_tickets = (team_jira_doc.to_dict() or {}).get("tickets", [])
#             merged_tickets = _merge_jira_tickets(existing_team_tickets, tickets)
#             team_sources_ref.document(team_jira_doc.id).update({
#                 "nickname": "jira",
#                 "tickets": merged_tickets,
#                 "updated_at": firestore.SERVER_TIMESTAMP
#             })
#         else:
#             team_sources_ref.add({
#                 "type": "jira",
#                 "nickname": "jira",
#                 "tickets": tickets,
#                 "created_at": firestore.SERVER_TIMESTAMP,
#                 "updated_at": firestore.SERVER_TIMESTAMP
#             })
        
#         ticket_texts = []
#         ticket_metadatas = []
#         vector_ids = []
#         for ticket in tickets:
#             ticket_id = str(ticket.get("id"))
#             text = _build_jira_ticket_text(ticket)
#             if not text:
#                 text = f"Jira Ticket ID {ticket_id}"
#             ticket_texts.append(text)
#             ticket_metadatas.append({
#                 "nickname": "jira",
#                 "source": "jira",
#                 "jira_id": ticket_id,
#                 "text_preview": text,
#             })
#             raw_id = f"jira_{team_id}_{ticket_id}"
#             safe_id = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in raw_id)
#             vector_ids.append(safe_id)

#         openai_client = get_openai_client()
#         embeddings = generate_embeddings(ticket_texts, openai_client)

#         vectors = []
#         for embedding, metadata, vector_id in zip(embeddings, ticket_metadatas, vector_ids):
#             vectors.append({
#                 "id": vector_id,
#                 "values": embedding,
#                 "metadata": metadata,
#             })

#         index_name = "production"
#         total_uploaded = upload_vectors_to_pinecone(vectors, namespace, index_name=index_name, batch_size=100)

#         return jsonify({
#             "status": "success",
#             "vectors_uploaded": total_uploaded,
#             "namespace": namespace,
#             "index": index_name,
#         }), 200
    
#     except KeyError as e:
#         logger.error(f"Jira add ticket error: {str(e)}")
#         return jsonify({"status": "failure", "error": str(e)}), 400
#     except Exception as e:
#         logger.error(f"Jira add ticket error: {str(e)}", exc_info=True)
#         return jsonify({"status": "failure", "error": f"Internal server error: {str(e)}"}), 500

@app.route("/api/jira/delete_ticket", methods=["POST"])
@require_solari_key
def jira_delete_ticket():
    """
    Delete a Jira ticket from both Firestore sources and Pinecone.
    
    Request Body:
    {
      "user_id": "firebase_uid",
      "agent_id": "agent_id",
      "jira_id": "10001",
      "namespace": "your-namespace"
    }
    """
    try:
        body = request.get_json(force=True) or {}
        user_id = body.get("user_id")
        agent_id = body.get("agent_id")
        jira_id = body.get("jira_id")
        
        if not user_id:
            return jsonify({"status": "failure", "error": "user_id is required"}), 400
        if not agent_id:
            return jsonify({"status": "failure", "error": "agent_id is required"}), 400
        if not jira_id:
            return jsonify({"status": "failure", "error": "jira_id is required"}), 400
        
        db = firestore.client()
        team_id = get_team_id_for_uid(db, user_id)
        namespace = _get_team_pinecone_namespace(db, team_id)
        
        removed_agent = False
        removed_team = False
        
        # --- Agent-level Jira source ---
        agent_sources_ref = (
            db.collection("teams").document(team_id)
              .collection("agents").document(agent_id)
              .collection("sources")
        )
        agent_jira_docs = agent_sources_ref.where("type", "==", "jira").stream()
        agent_jira_doc = next(agent_jira_docs, None)
        if agent_jira_doc:
            agent_data = agent_jira_doc.to_dict() or {}
            updated_tickets = _remove_jira_ticket_by_id(agent_data.get("tickets", []), jira_id)
            if len(updated_tickets) != len(agent_data.get("tickets", [])):
                removed_agent = True
            agent_sources_ref.document(agent_jira_doc.id).update({
                "tickets": updated_tickets,
                "updated_at": firestore.SERVER_TIMESTAMP
            })
        
        # --- Team-level Jira source ---
        team_sources_ref = db.collection("teams").document(team_id).collection("sources")
        team_jira_docs = team_sources_ref.where("type", "==", "jira").stream()
        team_jira_doc = next(team_jira_docs, None)
        if team_jira_doc:
            team_data = team_jira_doc.to_dict() or {}
            updated_tickets = _remove_jira_ticket_by_id(team_data.get("tickets", []), jira_id)
            if len(updated_tickets) != len(team_data.get("tickets", [])):
                removed_team = True
            team_sources_ref.document(team_jira_doc.id).update({
                "tickets": updated_tickets,
                "updated_at": firestore.SERVER_TIMESTAMP
            })
        
        # --- Pinecone delete ---
        raw_id = f"jira_{team_id}_{jira_id}"
        vector_id = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in raw_id)
        pc = get_pinecone_client()
        index = pc.Index("production")
        index.delete(ids=[vector_id], namespace=namespace)
        
        return jsonify({
            "status": "success",
            "removed_agent": removed_agent,
            "removed_team": removed_team,
            "deleted_vector_id": vector_id,
        }), 200
    
    except KeyError as e:
        logger.error(f"Jira delete ticket error: {str(e)}")
        return jsonify({"status": "failure", "error": str(e)}), 400
    except Exception as e:
        logger.error(f"Jira delete ticket error: {str(e)}", exc_info=True)
        return jsonify({"status": "failure", "error": f"Internal server error: {str(e)}"}), 500

@app.route('/api/firebase/test', methods=['GET'])
@require_solari_key
def test_firebase():
    """
    Test Firebase connection for Auth, Storage, and Firestore.
    
    Returns:
        JSON response with connection status for each service
    """
    try:
        results = test_firebase_connection()
        
        # Determine overall status
        all_success = all(
            result['status'] == 'success' 
            for result in results.values()
        )
        
        overall_status = 'success' if all_success else 'partial' if any(
            result['status'] == 'success' 
            for result in results.values()
        ) else 'error'
        
        return jsonify({
            'status': overall_status,
            'services': results,
            'message': 'Firebase connection test completed'
        }), 200 if all_success else 207  # 207 Multi-Status if partial success
        
    except Exception as e:
        logger.error(f"Error testing Firebase connection: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'Failed to test Firebase connection: {str(e)}'
        }), 500

@app.route('/api/example', methods=['GET', 'POST'])
@require_solari_key
def example_endpoint():
    """Example endpoint demonstrating request handling"""
    if request.method == 'GET':
        return jsonify({
            'message': 'This is a GET request',
            'method': 'GET'
        }), 200
    
    elif request.method == 'POST':
        data = request.get_json() or {}
        return jsonify({
            'message': 'This is a POST request',
            'method': 'POST',
            'received_data': data
        }), 201

@app.route('/api/team/invite_members', methods=['POST', 'OPTIONS'])
@require_solari_key
def invite_team_members():
    if request.method == "OPTIONS":
        return add_cors_headers(make_response()), 204

    try:
        data = request.get_json() or {}
        emails = data.get('emails')
        user_id = data.get('userId')
        team_id = data.get('teamId')

        if not emails or not isinstance(emails, list) or not user_id or not team_id:
            return jsonify({
                'success': False,
                'error': 'Missing required parameters: emails, userId, and teamId'
            }), 400
        creator_first_name, _, error = _get_team_creator_first_name(team_id, user_id)
        if error:
            return jsonify({"success": False, "error": error["message"]}), error["status"]

        db = firestore.client()
        team_snap = db.collection("teams").document(team_id).get()
        if not team_snap.exists:
            return jsonify({"success": False, "error": "team_not_found"}), 404
        team_data = team_snap.to_dict() or {}
        team_name = (team_data.get("team_name") or "").strip()
        invite_code = (team_data.get("invite_code") or "").strip()
        if not team_name or not invite_code:
            return jsonify({
                "success": False,
                "error": "team_name_or_invite_code_missing"
            }), 500

        inviter_snap = db.collection("users").document(str(user_id)).get()
        if not inviter_snap.exists:
            return jsonify({"success": False, "error": "inviter_not_found"}), 404
        inviter_data = inviter_snap.to_dict() or {}
        inviter_name = (inviter_data.get("displayName") or "").strip()
        if not inviter_name:
            return jsonify({"success": False, "error": "inviter_display_name_missing"}), 500

        subject = f"{creator_first_name} invited you to the {team_name} workspace"
        login_url = f"http://localhost:3000/login?invite={invite_code}"
        text_body = (
            f"Hey there! {inviter_name} has invited you to the {team_name} solari workspace!\n\n"
            "Join the team by logging in here, and using the invite code below to join your team.\n\n"
            f"{team_name}'s invite code: {invite_code}."
        )
        html_body = (
            f"Hey there! {inviter_name} has invited you to the {team_name} solari workspace!<br><br>"
            f'Join the team by logging in <a href="{login_url}">here</a>, '
            "and using the invite code below to join your team.<br><br>"
            f"{team_name}'s invite code: {invite_code}."
        )

        bounced_emails = []
        for email in emails:
            response = send_email(email, subject, text_body, html_body=html_body)
            if not response:
                bounced_emails.append(email)
                continue
            if response.status_code != 200:
                bounced_emails.append(email)

        if bounced_emails:
            return jsonify({
                'success': False,
                'error': 'Failed to send one or more invitations',
                'bounced_emails': bounced_emails
            }), 500

        return jsonify({'status': 'ok'}), 200

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/user/send_welcome_email', methods=['POST', 'OPTIONS'])
@require_solari_key
def send_welcome_email():
    if request.method == "OPTIONS":
        return add_cors_headers(make_response()), 204

    try:
        data = request.get_json() or {}
        user_id = data.get("userId") or data.get("user_id")
        if not user_id:
            return jsonify({"success": False, "error": "missing_user_id"}), 400

        db = firestore.client()
        user_snap = db.collection("users").document(str(user_id)).get()
        if not user_snap.exists:
            return jsonify({"success": False, "error": "user_not_found"}), 404

        user_data = user_snap.to_dict() or {}
        to_email = (user_data.get("email") or "").strip()
        if not to_email:
            return jsonify({"success": False, "error": "user_email_missing"}), 400

        subject, text_body, html_body = build_signup_welcome_email(user_data)
        response = send_email(to_email, subject, text_body, html_body=html_body or None)
        if not response or response.status_code != 200:
            return jsonify({"success": False, "error": "email_send_failed"}), 500

        return jsonify({"success": True}), 200
    except Exception as e:
        logger.error(f"Error sending welcome email: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/agent/add_members', methods=['POST', 'OPTIONS'])
@require_solari_key
def add_agent_members():
    if request.method == "OPTIONS":
        return add_cors_headers(make_response()), 204

    try:
        data = request.get_json() or {}
        team_id = data.get("team_id") or data.get("teamId")
        agent_id = data.get("agent_id") or data.get("agentId")
        agent_name = data.get("agent_name") or data.get("agentName")
        members = data.get("members")

        if not team_id or not agent_id or not agent_name or not isinstance(members, list) or not members:
            return jsonify({
                "success": False,
                "error": "Missing required parameters: team_id, agent_id, agent_name, members"
            }), 400

        db = firestore.client()
        team_users_ref = db.collection("teams").document(team_id).collection("users")
        members_ref = (
            db.collection("teams").document(team_id)
              .collection("agents").document(agent_id)
              .collection("agent-members")
        )

        successes = []
        failures = []

        for member in members:
            email_raw = (member or {}).get("email")
            permission_raw = (member or {}).get("permission")
            if not email_raw or not permission_raw:
                failures.append({
                    "email": email_raw or "",
                    "error": "missing_email_or_permission"
                })
                continue

            permission = str(permission_raw).strip().lower()
            if permission not in ("view", "edit", "admin"):
                failures.append({
                    "email": email_raw,
                    "error": "invalid_permission"
                })
                continue

            email = str(email_raw).strip()
            if not email:
                failures.append({
                    "email": email_raw or "",
                    "error": "invalid_email"
                })
                continue

            try:
                # Find team user by email (expecting one match)
                user_docs = list(team_users_ref.where("email", "==", email).limit(1).stream())
                if not user_docs and email.lower() != email:
                    user_docs = list(team_users_ref.where("email", "==", email.lower()).limit(1).stream())

                if not user_docs:
                    failures.append({
                        "email": email,
                        "error": "user_not_found_in_team"
                    })
                    continue

                user_doc = user_docs[0]
                user_id = user_doc.id
                user_data = user_doc.to_dict() or {}
                user_email = user_data.get("email") or email
                display_name = user_data.get("displayName") or ""
                photo_url = user_data.get("photoURL") or ""

                team_user_ref = team_users_ref.document(user_id)
                member_doc_ref = members_ref.document(user_id)

                @firestore.transactional
                def update_member(txn: firestore.Transaction):
                    team_user_snap = team_user_ref.get(transaction=txn)
                    if not team_user_snap.exists:
                        return {"ok": False, "error": "team_user_not_found"}

                    team_user_data = team_user_snap.to_dict() or {}
                    existing_agents = team_user_data.get("agents") or []
                    if not isinstance(existing_agents, list):
                        existing_agents = []

                    updated_agents = []
                    agent_entry_found = False
                    existing_agent_permission = None
                    for entry in existing_agents:
                        if isinstance(entry, dict) and entry.get("agent_id") == agent_id:
                            agent_entry_found = True
                            existing_agent_permission = entry.get("permission")
                            updated_entry = dict(entry)
                            if permission != existing_agent_permission:
                                updated_entry["permission"] = permission
                            updated_entry["agent_name"] = agent_name
                            updated_agents.append(updated_entry)
                        else:
                            updated_agents.append(entry)

                    if not agent_entry_found:
                        updated_agents.append({
                            "agent_id": agent_id,
                            "role": permission,
                            "agent_name": agent_name
                        })

                    member_snap = member_doc_ref.get(transaction=txn)
                    member_existing = member_snap.to_dict() if member_snap.exists else {}
                    member_existing_permission = member_existing.get("permission")

                    txn.update(team_user_ref, {"agents": updated_agents})
                    txn.set(member_doc_ref, {
                        "uid": user_id,
                        "email": user_email,
                        "role": permission,
                        "displayName": display_name,
                        "photoURL": photo_url
                    }, merge=True)

                    no_agent_change = agent_entry_found and existing_agent_permission == permission
                    no_member_change = member_snap.exists and member_existing_permission == permission

                    return {
                        "ok": True,
                        "already_added": no_agent_change and no_member_change
                    }

                txn = db.transaction()
                result = update_member(txn)
                if not result.get("ok"):
                    failures.append({
                        "email": email,
                        "error": result.get("error") or "update_failed"
                    })
                    continue

                if result.get("already_added"):
                    successes.append({
                        "email": user_email,
                        "uid": user_id,
                        "role": permission,
                        "message": "member was already added to team"
                    })
                else:
                    successes.append({
                        "email": user_email,
                        "uid": user_id,
                        "role": permission,
                        "message": "member added"
                    })

            except Exception as e:
                logger.error(f"Error adding member {email}: {str(e)}", exc_info=True)
                failures.append({
                    "email": email,
                    "error": str(e)
                })

        return jsonify({
            "success": True,
            "added": successes,
            "failures": failures
        }), 200

    except Exception as e:
        logger.error(f"Error in add_agent_members: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/agent/update_model', methods=['POST'])
@require_solari_key
def update_agent_model():
    """
    Update the model_provider field for an agent.
    
    Expected request body:
    {
        "user_id": "user123",  # or "userid"
        "agent_id": "agent123",  # or "agentId"
        "model_provider": "gpt-4o-mini"
    }
    """
    try:
        data = request.get_json() or {}
        user_id = data.get("user_id") or data.get("userid")
        agent_id = data.get("agent_id") or data.get("agentId")
        model_provider = data.get("model_provider")

        if not user_id or not agent_id or not model_provider:
            return jsonify({
                "success": False,
                "error": "Missing required parameters: user_id, agent_id, model_provider"
            }), 400

        db = firestore.client()
        try:
            team_id = get_team_id_for_uid(db, user_id)
        except KeyError as e:
            return jsonify({
                "success": False,
                "error": str(e)
            }), 404

        agent_ref = (
            db.collection("teams").document(team_id)
              .collection("agents").document(agent_id)
        )
        agent_snap = agent_ref.get()
        if not agent_snap.exists:
            return jsonify({
                "success": False,
                "error": "agent_not_found"
            }), 404

        agent_ref.update({
            "model_provider": model_provider
        })

        return jsonify({
            "success": True,
            "agent_id": agent_id,
            "model_provider": model_provider
        }), 200
    except Exception as e:
        logger.error(f"Error updating agent model: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/agent/remove_members', methods=['POST', 'OPTIONS'])
@require_solari_key
def remove_agent_member():
    if request.method == "OPTIONS":
        return add_cors_headers(make_response()), 204

    try:
        data = request.get_json() or {}
        team_id = data.get("team_id") or data.get("teamId")
        agent_id = data.get("agent_id") or data.get("agentId")
        user_id = data.get("user_id") or data.get("userId")

        if not team_id or not agent_id or not user_id:
            return jsonify({
                "success": False,
                "error": "Missing required parameters: team_id, agent_id, user_id"
            }), 400

        db = firestore.client()
        team_user_ref = db.collection("teams").document(team_id).collection("users").document(user_id)
        member_doc_ref = (
            db.collection("teams").document(team_id)
              .collection("agents").document(agent_id)
              .collection("agent-members").document(user_id)
        )

        @firestore.transactional
        def remove_member(txn: firestore.Transaction):
            team_user_snap = team_user_ref.get(transaction=txn)
            if not team_user_snap.exists:
                return {"ok": False, "error": "team_user_not_found"}

            team_user_data = team_user_snap.to_dict() or {}
            existing_agents = team_user_data.get("agents") or []
            if not isinstance(existing_agents, list):
                existing_agents = []

            updated_agents = []
            for entry in existing_agents:
                if isinstance(entry, dict):
                    entry_agent_id = entry.get("agent_id") or entry.get("agentId")
                    if entry_agent_id == agent_id:
                        continue
                elif isinstance(entry, str):
                    if entry == agent_id:
                        continue
                updated_agents.append(entry)

            txn.update(team_user_ref, {"agents": updated_agents})
            txn.delete(member_doc_ref)

            return {"ok": True}

        txn = db.transaction()
        result = remove_member(txn)
        if not result.get("ok"):
            return jsonify({
                "success": False,
                "error": result.get("error") or "remove_failed"
            }), 400

        return jsonify({"success": True}), 200

    except Exception as e:
        logger.error(f"Error in remove_agent_member: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/agent/list_members', methods=['POST', 'OPTIONS'])
@require_solari_key
def list_agent_members():
    if request.method == "OPTIONS":
        return add_cors_headers(make_response()), 204

    try:
        data = request.get_json() or {}
        team_id = data.get("team_id") or data.get("teamId")
        agent_id = data.get("agent_id") or data.get("agentId")

        if not team_id or not agent_id:
            return jsonify({
                "success": False,
                "error": "Missing required parameters: team_id, agent_id"
            }), 400

        db = firestore.client()
        members_ref = (
            db.collection("teams").document(team_id)
              .collection("agents").document(agent_id)
              .collection("agent-members")
        )

        members = []
        for doc in members_ref.stream():
            payload = doc.to_dict() or {}
            payload["id"] = doc.id
            members.append(payload)

        return jsonify({
            "success": True,
            "members": members
        }), 200

    except Exception as e:
        logger.error(f"Error in list_agent_members: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/agent/delete', methods=['POST'])
@require_solari_key
def delete_agent():
    """
    Delete an agent (and all subcollections) if user has admin permissions.
    
    Request Body:
    {
      "user_id": "firebase_uid",
      "agent_id": "agent_id"
    }
    """
    try:
        data = request.get_json() or {}
        user_id = data.get("user_id") or data.get("userId")
        agent_id = data.get("agent_id") or data.get("agentId")

        if not user_id or not agent_id:
            return jsonify({
                "status": "failure",
                "reason": "missing_user_id_or_agent_id"
            }), 400

        db = firestore.client()
        team_id = get_team_id_for_uid(db, user_id)

        team_user_ref = db.collection("teams").document(team_id).collection("users").document(user_id)
        team_user_snap = team_user_ref.get()
        if not team_user_snap.exists:
            return jsonify({
                "status": "failure",
                "reason": "team_user_not_found"
            }), 404

        team_role = (team_user_snap.to_dict() or {}).get("role")
        if team_role == "admin":
            has_permission = True
            permission_used = "team_admin"
        else:
            member_ref = (
                db.collection("teams").document(team_id)
                  .collection("agents").document(agent_id)
                  .collection("agent-members").document(user_id)
            )
            member_snap = member_ref.get()
            member_role = (member_snap.to_dict() or {}).get("role") if member_snap.exists else None
            has_permission = member_role == "admin"
            permission_used = member_role or "none"

        if not has_permission:
            return jsonify({
                "status": "failure",
                "reason": f"permission error {permission_used}"
            }), 403

        agent_ref = db.collection("teams").document(team_id).collection("agents").document(agent_id)
        agent_snap = agent_ref.get()
        if not agent_snap.exists:
            return jsonify({
                "status": "failure",
                "reason": "agent_not_found"
            }), 404

        agent_name = (agent_snap.to_dict() or {}).get("name") or ""
        _delete_firestore_document_recursive(agent_ref)

        return jsonify({
            "status": "success",
            "agent_deleted": agent_name
        }), 200

    except Exception as e:
        logger.error(f"Error in delete_agent: {str(e)}", exc_info=True)
        return jsonify({"status": "failure", "reason": str(e)}), 500

# @app.route('/api/pinecone_doc_upload', methods=['POST'])
# @require_solari_key
# def pinecone_doc_upload():
#     """
#     Upload documents to Pinecone after processing.
    
#     Expected request body:
#     {
#         "namespace": "your-namespace",
#         "file_path": "users/userId/files/file.pdf",
#         "nickname": "optional-nickname"  # optional
#     }
    
#     Process:
#     1. Download file from Firebase Storage
#     2. Extract text from file (PDF/DOCX)
#     3. Chunk the text
#     4. Generate embeddings using OpenAI
#     5. Upload to Pinecone
    
#     Returns:
#         JSON response with upload status
#     """
#     try:
#         # Get request data
#         data = request.get_json()
        
#         if not data:
#             return jsonify({
#                 'status': 'error',
#                 'message': 'Request body is required'
#             }), 400
        
#         # Validate required fields
#         namespace = data.get('namespace')
#         if not namespace:
#             return jsonify({
#                 'status': 'error',
#                 'message': 'namespace parameter is required'
#             }), 400
        
#         file_path = data.get('file_path')
#         if not file_path:
#             return jsonify({
#                 'status': 'error',
#                 'message': 'file_path parameter is required'
#             }), 400
        
#         # Optional parameter
#         nickname = data.get('nickname', '')
        
#         logger.info(f"Processing file: {file_path} for namespace: {namespace}")
        
#         # Step 1: Download file from Firebase Storage
#         logger.info("Downloading file from Firebase Storage...")
#         file_content = download_file_from_firebase(file_path)
        
#         # Step 2: Extract text from file
#         logger.info("Extracting text from file...")
#         text = extract_text_from_file(file_content, file_path)
        
#         if not text or len(text.strip()) == 0:
#             return jsonify({
#                 'status': 'error',
#                 'message': 'No text could be extracted from the file'
#             }), 400
        
#         logger.info(f"Extracted {len(text)} characters of text")
        
#         # Step 3: Chunk the text
#         logger.info("Chunking text...")
#         text_chunks = chunk_text(text, chunk_size=1000, chunk_overlap=200)
#         logger.info(f"Created {len(text_chunks)} chunks")
        
#         # Step 4: Generate embeddings
#         logger.info("Generating embeddings...")
#         openai_client = get_openai_client()
#         embeddings = generate_embeddings(text_chunks, openai_client)
        
#         # Step 5: Prepare vectors for Pinecone
#         # Use file_path as base for IDs (sanitize it)
#         file_id_base = file_path.replace('/', '_').replace(' ', '_')
#         vectors = []
#         for i, (chunk, embedding) in enumerate(zip(text_chunks, embeddings)):
#             vectors.append({
#                 'id': f"{file_id_base}_chunk_{i}",
#                 'values': embedding,
#                 'metadata': {
#                     'file_path': file_path,
#                     'chunk_index': i,
#                     'text_preview': chunk[:500],  # Store first 500 chars as metadata
#                     'nickname': nickname  # Add nickname to metadata
#                 }
#             })
        
#         # Step 6: Upload to Pinecone in batches
#         logger.info(f"Uploading {len(vectors)} vectors to Pinecone in batches...")
#         index_name = 'production'
#         total_uploaded = upload_vectors_to_pinecone(vectors, namespace, index_name=index_name, batch_size=100)
        
#         logger.info(f"Successfully uploaded {total_uploaded} vectors to namespace '{namespace}'")
        
#         return jsonify({
#             'status': 'success',
#             'message': f'Successfully processed and uploaded {len(vectors)} vectors to namespace "{namespace}"',
#             'namespace': namespace,
#             'index': index_name,
#             'file_path': file_path,
#             'vectors_uploaded': len(vectors),
#             'chunks_created': len(text_chunks),
#             'text_length': len(text),
#             'nickname': nickname
#         }), 200
        
#     except ValueError as e:
#         logger.error(f"Validation error: {str(e)}")
#         return jsonify({
#             'status': 'error',
#             'message': str(e)
#         }), 400
#     except Exception as e:
#         logger.error(f"Error processing file: {str(e)}", exc_info=True)
#         return jsonify({
#             'status': 'error',
#             'message': f'Failed to process file: {str(e)}'
#         }), 500

@app.route("/api/pinecone_doc_upload", methods=["POST"])
@require_solari_key
def pinecone_doc_upload():
    """
    Enqueue doc ingestion job.

    Request Body:
    {
      "user_id": "...",
      "agent_id": "...",
      "file_path": "teams/{teamId}/sources/{sourceId}/.../file.pdf" OR "users/{uid}/.../file.pdf",
      "nickname": "optional"
    }
    # namespace is derived from the team doc
    """
    body = request.get_json(force=True) or {}
    user_id = body.get("user_id") or body.get("uid")
    agent_id = body.get("agent_id") or body.get("agentId")
    file_path = body.get("file_path") or body.get("filePath")
    nickname = (body.get("nickname") or "").strip()

    if not user_id:
        return jsonify({"status": "failure", "error": "user_id is required"}), 400
    if not agent_id:
        return jsonify({"status": "failure", "error": "agent_id is required"}), 400
    if not file_path:
        return jsonify({"status": "failure", "error": "file_path is required"}), 400

    db = firestore.client()
    team_id = get_team_id_for_uid(db, user_id)
    namespace = _get_team_pinecone_namespace(db, team_id)

    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(days=30)

    job_id = uuid.uuid4().hex
    job_ref = (
        db.collection("teams").document(team_id)
          .collection("upload_jobs").document(job_id)
    )

    # single “source” for this upload
    source_key = f"doc:{file_path}"

    job_ref.set({
        "job_type": "ingest_sources",
        "connector": "doc",
        "status": "queued",
        "created_at": firestore.SERVER_TIMESTAMP,
        "updated_at": firestore.SERVER_TIMESTAMP,
        "expires_at": expires_at,
        "namespace": namespace,

        "locked_by": None,
        "locked_until": None,
        "progress": 0,
        "message": "Queued",
        "created_by_user_id": user_id,

        "team_id": team_id,
        "agent_id": agent_id,

        "sources": [{
            "source_key": source_key,
            "type": "doc",
            "id": file_path,          # keep it simple: id == file_path
            "file_path": file_path,
            "nickname": nickname or None,
            "namespace": namespace,

            "status": "queued",
            "stage": "queued",
            "checkpoint": {
                "chunk_index": 0,
                "total_chunks": None,
            },
            "error": None,
            "updated_at": now,         # datetime, not SERVER_TIMESTAMP
        }],
    })

    return jsonify({
        "status": "success",
        "job_id": job_id,
        "team_id": team_id,
        "queued_sources": 1,
    }), 200

# Add this function after the generate_embeddings function (around line 354)

def scrape_website_with_firecrawl(url: str) -> dict:
    """
    Scrape a website using Firecrawl and return markdown content.
    
    Args:
        url: The website URL to scrape
    
    Returns:
        Dictionary containing markdown content and metadata
    
    Raises:
        ValueError: If scraping fails or returns no content
    """
    if firecrawl is None:
        raise ValueError("Firecrawl is not initialized. Please set FIRECRAWL_API_KEY environment variable.")
    
    try:
        logger.info(f"Scraping website: {url}")
        result = firecrawl.scrape(url, formats=["markdown"])
        
        # Handle Document object - access attributes directly
        if hasattr(result, 'markdown'):
            # It's a Document object
            markdown = result.markdown or ""
            metadata = result.metadata if hasattr(result, 'metadata') else {}
        elif isinstance(result, dict):
            # Fallback: handle dict response (if API changes)
            if not result.get("success"):
                raise ValueError(f"Firecrawl scraping failed for URL: {url}")
            data = result.get("data", {})
            markdown = data.get("markdown", "")
            metadata = data.get("metadata", {})
        else:
            raise ValueError(f"Unexpected response type from Firecrawl: {type(result)}")
        
        if not markdown or len(markdown.strip()) == 0:
            raise ValueError(f"No markdown content extracted from URL: {url}")
        
        logger.info(f"Successfully scraped {len(markdown)} characters of markdown from {url}")
        
        return {
            "markdown": markdown,
            "metadata": metadata if isinstance(metadata, dict) else {}
        }
    except Exception as e:
        logger.error(f"Error scraping website with Firecrawl: {str(e)}")
        raise ValueError(f"Failed to scrape website: {str(e)}")

@app.post("/api/website/add_urls")
@require_solari_key
def add_website_urls():
    """
    Add website URLs to upload_jobs for background processing.

    Request body:
    {
      "user_id": "...",
      "agent_id": "...",
      "sites": [
        { "url": "https://example.com", "nickname": "Example" }
      ]
    }
    """
    body = request.get_json(force=True) or {}

    user_id = body.get("user_id")
    agent_id = body.get("agent_id")
    sites = body.get("sites")

    if not user_id or not agent_id:
        return jsonify({"error": "user_id and agent_id are required"}), 400
    if not isinstance(sites, list) or not sites:
        return jsonify({"error": "sites must be a non-empty list"}), 400

    db = firestore.client()
    now = utcnow()

    try:
        team_id = get_team_id_for_uid(db, user_id)
    except Exception as e:
        return jsonify({"error": str(e)}), 404

    sources = []
    seen = set()

    for s in sites:
        if not isinstance(s, dict):
            continue

        url = (s.get("url") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)

        nickname = (s.get("nickname") or url).strip()

        sources.append({
            "type": "website",
            "id": url,
            "source_key": f"website:{url}",
            "title": url,
            "nickname": nickname,

            # worker-controlled fields
            "status": "queued",
            "stage": "queued",
            "error": None,
            "checkpoint": {},
        })

    if not sources:
        return jsonify({"error": "no valid sites"}), 400

    job_ref = (
        db.collection("teams")
          .document(team_id)
          .collection("upload_jobs")
          .document()
    )

    job_doc = {
        "job_type": "ingest_sources",
        "connector": "website",

        "team_id": team_id,
        "agent_id": agent_id,
        "created_by_user_id": user_id,

        "status": "queued",
        "message": "Queued",
        "progress": 0,

        "created_at": now,
        "updated_at": now,
        "locked_by": None,
        "locked_until": None,

        "sources": sources,
    }

    job_ref.set(job_doc)

    return jsonify({
        "status": "success",
        "job_id": job_ref.id,
        "team_id": team_id,
        "queued_sources": len(sources),
    }), 200

# @app.route('/api/pinecone_website_upload', methods=['POST'])
# @require_solari_key
# def pinecone_website_upload():
#     """
#     Upload website content to Pinecone after scraping and processing.
    
#    w
    
#     Process:
#     1. Scrape website URL with Firecrawl to get markdown
#     2. Chunk the markdown
#     3. Generate embeddings using OpenAI
#     4. Upload to Pinecone
    
#     Returns:
#         JSON response with upload status
#     """
#     try:
#         # Get request data
#         data = request.get_json()
        
#         if not data:
#             return jsonify({
#                 'status': 'error',
#                 'message': 'Request body is required'
#             }), 400
        
#         # Validate required fields
#         namespace = data.get('namespace')
#         if not namespace:
#             return jsonify({
#                 'status': 'error',
#                 'message': 'namespace parameter is required'
#             }), 400
        
#         url = data.get('url')
#         if not url:
#             return jsonify({
#                 'status': 'error',
#                 'message': 'url parameter is required'
#             }), 400
        
#         # Optional parameters with defaults
#         chunk_size = data.get('chunk_size', 1000)
#         chunk_overlap = data.get('chunk_overlap', 200)
#         nickname = data.get('nickname', '')
        
#         logger.info(f"Processing website: {url} for namespace: {namespace}")
        
#         # Step 1: Scrape website with Firecrawl
#         logger.info("Scraping website with Firecrawl...")
#         scrape_result = scrape_website_with_firecrawl(url)
#         markdown = scrape_result["markdown"]
#         website_metadata = scrape_result["metadata"]
        
#         if not markdown or len(markdown.strip()) == 0:
#             return jsonify({
#                 'status': 'error',
#                 'message': 'No content could be extracted from the website'
#             }), 400
        
#         logger.info(f"Extracted {len(markdown)} characters of markdown")
        
#         # Step 2: Chunk the markdown
#         logger.info("Chunking markdown...")
#         text_chunks = chunk_text(markdown, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
#         logger.info(f"Created {len(text_chunks)} chunks")
        
#         # Step 3: Generate embeddings
#         logger.info("Generating embeddings...")
#         openai_client = get_openai_client()
#         embeddings = generate_embeddings(text_chunks, openai_client)
        
#         # Step 4: Prepare vectors for Pinecone
#         # Sanitize URL for use in IDs
#         url_id_base = url.replace('https://', '').replace('http://', '').replace('/', '_').replace(' ', '_').replace('.', '_')
#         # Remove any remaining special characters that might cause issues
#         url_id_base = ''.join(c if c.isalnum() or c == '_' else '_' for c in url_id_base)
        
#         vectors = []
#         for i, (chunk, embedding) in enumerate(zip(text_chunks, embeddings)):
#             vectors.append({
#                 'id': f"{url_id_base}_chunk_{i}",
#                 'values': embedding,
#                 'metadata': {
#                     'url': url,
#                     'chunk_index': i,
#                     'text_preview': chunk[:500],  # Store first 500 chars as metadata
#                     'source': 'website',
#                     'nickname': nickname,  # Add nickname to metadata
#                     # Include relevant metadata from Firecrawl
#                     'title': website_metadata.get('title', ''),
#                     'description': website_metadata.get('description', ''),
#                     'sourceURL': website_metadata.get('sourceURL', url),
#                 }
#             })
        
#         # Step 5: Upload to Pinecone in batches
#         logger.info(f"Uploading {len(vectors)} vectors to Pinecone in batches...")
#         index_name = 'production'
#         total_uploaded = upload_vectors_to_pinecone(vectors, namespace, index_name=index_name, batch_size=100)
        
#         logger.info(f"Successfully uploaded {total_uploaded} vectors to namespace '{namespace}'")
        
#         return jsonify({
#             'status': 'success',
#             'message': f'Successfully processed and uploaded {len(vectors)} vectors to namespace "{namespace}"',
#             'namespace': namespace,
#             'index': index_name,
#             'url': url,
#             'vectors_uploaded': len(vectors),
#             'chunks_created': len(text_chunks),
#             'markdown_length': len(markdown),
#             'website_metadata': website_metadata,
#             'nickname': nickname
#         }), 200
        
#     except ValueError as e:
#         logger.error(f"Validation error: {str(e)}")
#         return jsonify({
#             'status': 'error',
#             'message': str(e)
#         }), 400
#     except Exception as e:
#         logger.error(f"Error processing website: {str(e)}", exc_info=True)
#         return jsonify({
#             'status': 'error',
#             'message': f'Failed to process website: {str(e)}'
#         }), 500

@app.route('/api/pinecone_slack_upload', methods=['POST'])
@require_solari_key
def pinecone_slack_upload():
    """
    Upload latest Slack transcript chunk to Pinecone.

    Expected request body:
    {
        "uid": "...",
        "agent_id": "...",
        "source_id": "...",   # slack channel id (optional if channel_id provided)
        "channel_id": "...",  # optional, defaults to source_id
        "channel_name": "...",  # optional if existing source has name
        "namespace": "...",
        "nickname": "...",  # optional
        "chunk_n": 20,     # optional
        "overlap_n": 5     # optional
    }
    """
    try:
        data = request.get_json() or {}

        uid = data.get("uid")
        agent_id = data.get("agent_id") or data.get("agentId")
        source_id = data.get("source_id") or data.get("sourceId")
        channel_id = data.get("channel_id") or data.get("channelId") or source_id
        channel_name = data.get("channel_name") or data.get("channelName")
        namespace = data.get("namespace")
        nickname = data.get("nickname", "")
        team_id = data.get("team_id") or data.get("teamId")

        if not uid or not agent_id or not channel_id or not namespace:
            return jsonify({
                "status": "error",
                "message": "uid, agent_id, channel_id (or source_id), and namespace are required"
            }), 400

        chunk_n = int(data.get("chunk_n", 20))
        overlap_n = int(data.get("overlap_n", 5))

        db = firestore.client()

        try:
            result = pinecone_slack_upload_internal(
                db=db,
                uid=uid,
                agent_id=agent_id,
                channel_id=channel_id,
                channel_name=channel_name,
                namespace=namespace,
                nickname=nickname,
                chunk_n=chunk_n,
                overlap_n=overlap_n,
                team_id=team_id,
                source_id=source_id or channel_id,
            )
        except KeyError as e:
            return jsonify({"status": "error", "message": str(e)}), 404
        except Exception as e:
            return jsonify({"status": "error", "message": f"failed_to_load_transcript: {str(e)}"}), 500

        if not result.get("ok"):
            error = result.get("error")
            status = 500
            if error in ("missing_channel_id_or_namespace", "channel_name_required"):
                status = 400
            elif error == "slack_api_error":
                status = 502
            elif error in ("no_messages_to_upload",):
                status = 400
            return jsonify({"status": "error", "message": error, "details": result.get("details")}), status

        result["status"] = "success"
        return jsonify(result), 200
    except Exception as e:
        logger.error(f"Error processing Slack transcript upload: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": f"failed_to_process_slack_transcript: {str(e)}"
        }), 500


@app.post("/api/slack/add_channels")  # feel free to keep your old route if you want
@require_solari_key
def slack_add_channels_enqueue():
    """
    Enqueue Slack ingestion job (v1.5).

    Request body:
    {
      "uid": "firebase_uid",
      "agent_id": "agent_id",
      "channels": [
        {"channel_id": "C123", "channel_name": "general", "nickname": "Company General"},
        {"channel_id": "C456", "channel_name": "sales", "nickname": ""}
      ]
    }

    Notes:
    - team_id is derived from uid (never accepted from client)
    - background worker will derive namespace from team_id
    - chunking params are worker constants
    """
    payload = request.get_json(silent=True) or {}

    uid = (payload.get("uid") or payload.get("user_id") or "").strip()
    agent_id = (payload.get("agent_id") or payload.get("agentId") or "").strip()
    channels = payload.get("channels") or []

    if not uid or not agent_id:
        return jsonify({"ok": False, "error": "missing_uid_or_agent_id"}), 400
    if not isinstance(channels, list) or not channels:
        return jsonify({"ok": False, "error": "missing_channels"}), 400

    db = firestore.client()
    now = utcnow()

    # ✅ Always derive team_id from uid
    try:
        team_id = get_team_id_for_uid(db, uid)
    except Exception as e:
        return jsonify({"ok": False, "error": "team_id_not_found", "details": str(e)}), 404

    # Normalize channels (dedupe within request)
    sources = []
    seen_channel_ids = set()

    for c in channels:
        if not isinstance(c, dict):
            continue

        channel_id = (c.get("channel_id") or c.get("channelId") or c.get("id") or "").strip()
        channel_name = (c.get("channel_name") or c.get("channelName") or c.get("name") or "").strip()
        nickname = (c.get("nickname") or "").strip()

        if not channel_id:
            continue
        if channel_id in seen_channel_ids:
            continue
        seen_channel_ids.add(channel_id)

        sources.append({
            "source_key": f"slack:{channel_id}",
            "type": "slack",
            "id": channel_id,
            "title": f"#{channel_name}" if channel_name else f"Slack {channel_id}",
            "channel_name": channel_name or None,
            "nickname": nickname or None,

            "status": "queued",
            "stage": "queued",
            "checkpoint": {
                # worker can optionally store paging / embed progress here later
                "last_synced_ts": None,
                "last_embedded_ts": None,
            },
            "error": None,
            "updated_at": now,
        })

    if not sources:
        return jsonify({"ok": False, "error": "no_valid_channels"}), 400

    now = utcnow()
    expires_at = now + timedelta(days=30)

    job_id = uuid.uuid4().hex
    job_ref = (
        db.collection("teams").document(team_id)
          .collection("upload_jobs").document(job_id)
    )

    job_ref.set({
        "job_type": "ingest_sources",
        "connector": "slack",
        "status": "queued",
        "created_at": firestore.SERVER_TIMESTAMP,
        "updated_at": firestore.SERVER_TIMESTAMP,
        "expires_at": expires_at,

        "locked_by": None,
        "locked_until": None,

        "progress": 0,
        "message": "Queued",

        "created_by_user_id": uid,
        "team_id": team_id,
        "agent_id": agent_id,

        "sources": sources,
    })

    return jsonify({
        "ok": True,
        "job_id": job_id,
        "team_id": team_id,
        "queued_sources": len(sources),
        "status": "queued",
    }), 200
# @app.post("/api/pinecone_slack_upload_batch")
# @require_solari_key
# def pinecone_slack_upload_batch_start():
#     """
#     Start a batch Pinecone upload for Slack channels.

#     Expected request body:
#     {
#         "uid": "...",
#         "agent_id": "...",
#         "namespace": "...",
#         "chunk_n": 20,     # optional
#         "overlap_n": 5,    # optional
#         "channels": [
#             {"channel_id": "...", "channel_name": "...", "nickname": "..." }
#         ]
#     }
#     """
#     payload = request.get_json(silent=True) or {}
#     uid = payload.get("uid")
#     agent_id = payload.get("agent_id") or payload.get("agentId")
#     namespace = payload.get("namespace")
#     channels = payload.get("channels") or []
#     chunk_n = int(payload.get("chunk_n", 20))
#     overlap_n = int(payload.get("overlap_n", 5))

#     if not uid or not agent_id or not namespace:
#         return jsonify({"ok": False, "error": "missing_uid_agent_id_or_namespace"}), 400
#     if not isinstance(channels, list) or len(channels) == 0:
#         return jsonify({"ok": False, "error": "missing_channels"}), 400

#     normalized = []
#     seen = set()
#     for c in channels:
#         if not isinstance(c, dict):
#             continue
#         channel_id = (c.get("channel_id") or c.get("channelId") or "").strip()
#         channel_name = (c.get("channel_name") or c.get("channelName") or "").strip()
#         nickname = (c.get("nickname") or "").strip()
#         if not channel_id or not channel_name:
#             continue
#         key = (channel_id, channel_name)
#         if key in seen:
#             continue
#         seen.add(key)
#         normalized.append({
#             "channel_id": channel_id,
#             "channel_name": channel_name,
#             "nickname": nickname,
#             "status": "queued",
#             "started_at": None,
#             "finished_at": None,
#             "result": None,
#             "error": None,
#         })

#     if not normalized:
#         return jsonify({"ok": False, "error": "no_valid_channels"}), 400

#     batch_id = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ") + "_" + uuid.uuid4().hex[:8]
#     now_unix = int(time.time())

#     try:
#         team_user_ref, _ = get_team_user_ref(db, uid)
#     except KeyError as e:
#         return jsonify({"ok": False, "error": str(e)}), 404

#     batch_ref = (
#         team_user_ref.collection("agents").document(agent_id)
#           .collection("slack_pinecone_batches").document(batch_id)
#     )

#     doc = {
#         "batch_id": batch_id,
#         "uid": uid,
#         "agent_id": agent_id,
#         "provider": "slack",
#         "status": "running",
#         "created_at": firestore.SERVER_TIMESTAMP,
#         "started_at_unix": now_unix,
#         "finished_at_unix": None,

#         "namespace": namespace,
#         "chunk_n": chunk_n,
#         "overlap_n": overlap_n,

#         "total": len(normalized),
#         "completed": 0,
#         "failed": 0,

#         "queue": normalized,
#         "cursor": 0,
#         "last_tick_at_unix": None,
#     }

#     try:
#         batch_ref.set(doc)
#     except Exception as e:
#         return jsonify({"ok": False, "error": "firestore_write_failed", "details": str(e)}), 500

#     return jsonify({
#         "ok": True,
#         "batch_id": batch_id,
#         "total": len(normalized),
#         "status": "running",
#         "next": {
#             "tick_endpoint": "/api/pinecone_slack_upload_batch/tick",
#             "status_endpoint": "/api/pinecone_slack_upload_batch/status",
#         }
#     })

@app.get("/api/pinecone_slack_upload_batch/status")
@require_solari_key
def pinecone_slack_upload_batch_status():
    uid = request.args.get("uid")
    agent_id = request.args.get("agent_id")
    batch_id = request.args.get("batch_id")

    if not uid or not agent_id or not batch_id:
        return jsonify({"ok": False, "error": "missing_uid_agent_id_or_batch_id"}), 400

    try:
        team_user_ref, _ = get_team_user_ref(db, uid)
    except KeyError as e:
        return jsonify({"ok": False, "error": str(e)}), 404

    batch_ref = (
        team_user_ref.collection("agents").document(agent_id)
          .collection("slack_pinecone_batches").document(batch_id)
    )

    snap = batch_ref.get()
    if not snap.exists:
        return jsonify({"ok": False, "error": "batch_not_found"}), 404

    doc = snap.to_dict() or {}
    return jsonify({
        "ok": True,
        "batch_id": batch_id,
        "uid": uid,
        "agent_id": agent_id,
        "status": doc.get("status"),
        "total": doc.get("total", 0),
        "completed": doc.get("completed", 0),
        "failed": doc.get("failed", 0),
        "cursor": doc.get("cursor", 0),
        "queue": doc.get("queue", []),
        "last_tick_at_unix": doc.get("last_tick_at_unix"),
        "finished_at_unix": doc.get("finished_at_unix"),
    })

@app.get("/api/pinecone_slack_upload_batch/tick")
@require_solari_key
def pinecone_slack_upload_batch_tick():
    uid = request.args.get("uid")
    agent_id = request.args.get("agent_id")
    batch_id = request.args.get("batch_id")

    if not uid or not agent_id or not batch_id:
        return jsonify({"ok": False, "error": "missing_uid_agent_id_or_batch_id"}), 400

    try:
        team_user_ref, team_id_solari = get_team_user_ref(db, uid)
    except KeyError as e:
        return jsonify({"ok": False, "error": str(e)}), 404

    batch_ref = (
        team_user_ref.collection("agents").document(agent_id)
          .collection("slack_pinecone_batches").document(batch_id)
    )

    now_unix = int(time.time())

    @firestore.transactional
    def claim_work(txn: firestore.Transaction):
        snap = batch_ref.get(transaction=txn)
        if not snap.exists:
            return {"ok": False, "error": "batch_not_found"}

        doc = snap.to_dict() or {}
        status = doc.get("status")
        if status not in ("running",):
            return {"ok": False, "error": "batch_not_running", "status": status}

        queue = doc.get("queue", []) or []
        cursor = int(doc.get("cursor", 0) or 0)

        if cursor >= len(queue):
            txn.update(batch_ref, {
                "status": "done",
                "finished_at_unix": doc.get("finished_at_unix") or now_unix,
                "last_tick_at_unix": now_unix,
            })
            return {"ok": True, "done": True, "doc": doc}

        item = queue[cursor] or {}
        if item.get("status") in ("done", "error", "skipped"):
            txn.update(batch_ref, {
                "cursor": cursor + 1,
                "last_tick_at_unix": now_unix,
            })
            return {"ok": True, "skipped_cursor_advance": True, "doc": doc}

        item["status"] = "running"
        item["started_at_unix"] = now_unix
        queue[cursor] = item

        txn.update(batch_ref, {
            "queue": queue,
            "last_tick_at_unix": now_unix,
        })

        return {
            "ok": True,
            "done": False,
            "cursor": cursor,
            "item": item,
            "doc": doc,
        }

    txn = db.transaction()
    claim = claim_work(txn)

    if not claim.get("ok"):
        return jsonify(claim), (404 if claim.get("error") == "batch_not_found" else 400)

    if claim.get("done"):
        doc = claim.get("doc") or {}
        return jsonify({
            "ok": True,
            "batch_id": batch_id,
            "status": "done",
            "total": doc.get("total", 0),
            "completed": doc.get("completed", 0),
            "failed": doc.get("failed", 0),
        })

    if claim.get("skipped_cursor_advance"):
        doc = claim.get("doc") or {}
        return jsonify({
            "ok": True,
            "batch_id": batch_id,
            "status": doc.get("status"),
            "cursor": doc.get("cursor", 0),
            "total": doc.get("total", 0),
        })

    item = claim.get("item") or {}
    cursor = claim.get("cursor", 0)
    namespace = (claim.get("doc") or {}).get("namespace")
    chunk_n = int((claim.get("doc") or {}).get("chunk_n", 20))
    overlap_n = int((claim.get("doc") or {}).get("overlap_n", 5))

    channel_id = item.get("channel_id")
    channel_name = item.get("channel_name")
    nickname = item.get("nickname") or ""

    result = pinecone_slack_upload_internal(
        db=db,
        uid=uid,
        agent_id=agent_id,
        channel_id=channel_id,
        channel_name=channel_name,
        namespace=namespace,
        nickname=nickname,
        chunk_n=chunk_n,
        overlap_n=overlap_n,
        team_id=None,
        source_id=channel_id,
    )

    queue_updates = {}
    status = "done" if result.get("ok") else "error"
    queue_updates["status"] = status
    queue_updates["finished_at_unix"] = int(time.time())
    queue_updates["result"] = result if result.get("ok") else None
    queue_updates["error"] = None if result.get("ok") else result.get("error")

    @firestore.transactional
    def finish_work(txn: firestore.Transaction):
        snap = batch_ref.get(transaction=txn)
        if not snap.exists:
            return {"ok": False, "error": "batch_not_found"}

        doc = snap.to_dict() or {}
        queue = doc.get("queue", []) or []
        if cursor >= len(queue):
            return {"ok": False, "error": "cursor_out_of_range"}

        queue[cursor] = {**queue[cursor], **queue_updates}
        completed = int(doc.get("completed", 0) or 0)
        failed = int(doc.get("failed", 0) or 0)
        if status == "done":
            completed += 1
        else:
            failed += 1

        txn.update(batch_ref, {
            "queue": queue,
            "cursor": cursor + 1,
            "completed": completed,
            "failed": failed,
            "last_tick_at_unix": int(time.time()),
        })
        return {"ok": True}

    finish_result = finish_work(db.transaction())
    if not finish_result.get("ok"):
        return jsonify(finish_result), 500

    return jsonify({
        "ok": True,
        "batch_id": batch_id,
        "cursor": cursor,
        "item_status": status,
        "result": result,
    })

def analyze_tabular_data(file_content: bytes, file_path: str) -> dict:
    """
    Analyze tabular data file (Excel, CSV, or TSV) to get row count and column types.
    
    Args:
        file_content: File content as bytes
        file_path: File path (used to determine file type)
    
    Returns:
        Dictionary with row_count and columns information
    """
    try:
        file_ext = file_path.lower().split('.')[-1]
        file_io = io.BytesIO(file_content)
        
        # Read file based on extension
        if file_ext in ['xlsx', 'xls']:
            df = pd.read_excel(file_io, engine='openpyxl')
        elif file_ext == 'csv':
            df = pd.read_csv(file_io)
        elif file_ext == 'tsv':
            df = pd.read_csv(file_io, sep='\t')
        else:
            raise ValueError(f"Unsupported file type for tabular analysis: {file_ext}")
        
        # Count rows (excluding header)
        row_count = len(df)
        
        # Get column information
        columns = {}
        for col_name, dtype in df.dtypes.items():
            # Convert pandas dtype to string representation
            dtype_str = str(dtype)
            # Map common pandas dtypes to more readable types
            if dtype_str.startswith('int'):
                type_name = 'integer'
            elif dtype_str.startswith('float'):
                type_name = 'float'
            elif dtype_str.startswith('bool'):
                type_name = 'boolean'
            elif dtype_str.startswith('datetime'):
                type_name = 'datetime'
            else:
                type_name = 'string'
            
            # Extract 3 example values (non-null, unique)
            example_values = []
            col_data = df[col_name].dropna()  # Remove null values
            
            if len(col_data) > 0:
                # Get unique values, then take first 3
                unique_values = col_data.unique()[:3]
                for val in unique_values:
                    # Convert to JSON-serializable type
                    if isinstance(val, pd.Timestamp):
                        # Convert datetime to ISO format string
                        example_values.append(val.isoformat())
                    elif isinstance(val, (int, float, bool, str)):
                        # Native Python types are JSON-serializable
                        example_values.append(val)
                    elif hasattr(val, 'item'):
                        # Convert numpy/pandas scalar types to native Python
                        example_values.append(val.item())
                    else:
                        # Fallback to string representation
                        example_values.append(str(val))
            
            columns[col_name] = {
                'type': type_name,
                'pandas_dtype': dtype_str,
                'example_values': example_values
            }
        
        logger.info(f"Analyzed tabular file: {row_count} rows, {len(columns)} columns")
        
        return {
            'row_count': row_count,
            'columns': columns
        }
    except Exception as e:
        logger.error(f"Error analyzing tabular data: {str(e)}")
        raise ValueError(f"Failed to analyze tabular data: {str(e)}")

def download_table_from_source(team_id: str, user_id: str, agent_id: str, document_id: str) -> dict:
    """
    Download a table file (CSV, Excel, etc.) from Firebase Storage using a Firestore source document
    and return both the file content and metadata.
    
    Args:
        team_id: Team identifier
        user_id: User identifier
        agent_id: Agent identifier
        document_id: Source document identifier
    
    Returns:
        Dictionary with:
            - file_content: File content as bytes
            - file_path: Path to the file in Firebase Storage
            - row_count: Number of rows in the table
            - columns: Dictionary of column names with their types
            - column_count: Number of columns
    
    Raises:
        ValueError: If document not found, filePath missing, or download/analysis fails
    """
    try:
        logger.info(f"Downloading table for team: {team_id}, user: {user_id}, agent: {agent_id}, document: {document_id}")
        
        # Step 1: Get document from Firestore
        db = firestore.client()
        doc_ref = (
            db.collection('teams').document(team_id)
              .collection('agents').document(agent_id)
              .collection('sources').document(document_id)
        )
        doc = doc_ref.get()
        
        if not doc.exists:
            raise ValueError(f'Document not found: {document_id}')
        
        doc_data = doc.to_dict()
        
        # Step 2: Extract filePath
        file_path = doc_data.get('filePath')
        if not file_path:
            raise ValueError('filePath not found in document')
        
        logger.info(f"Found filePath: {file_path}")
        
        # Step 3: Download file from Firebase Storage
        logger.info("Downloading file from Firebase Storage...")
        file_content = download_file_from_firebase(file_path)
        
        logger.info(f"Successfully downloaded file: {file_path} ({len(file_content)} bytes)")
        
        # Step 4: Analyze tabular data to get metadata
        logger.info("Analyzing tabular data...")
        analysis_result = analyze_tabular_data(file_content, file_path)
        
        logger.info(f"Analysis complete: {analysis_result['row_count']} rows, {len(analysis_result['columns'])} columns")
        
        # Step 5: Update Firestore document with metadata
        logger.info("Updating Firestore document with metadata...")
        doc_ref.update({
            'row_count': analysis_result['row_count'],
            'column_count': len(analysis_result['columns'])
        })
        logger.info("Successfully updated document with row_count and column_count")
        
        # Return both file content and metadata
        return {
            'file_content': file_content,
            'file_path': file_path,
            'row_count': analysis_result['row_count'],
            'columns': analysis_result['columns'],
            'column_count': len(analysis_result['columns'])
        }
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error downloading table: {str(e)}", exc_info=True)
        raise ValueError(f"Failed to download table: {str(e)}")

def materialize_table_file_atomic(file_bytes: bytes, user_id: str, file_id: str, file_path: str) -> str:
    """
    Save table file bytes to a temporary directory atomically.
    
    Args:
        file_bytes: File content as bytes
        user_id: User identifier (for unique file naming)
        file_id: File/document identifier (for unique file naming)
        file_path: Original file path (used to determine file extension)
    
    Returns:
        Path to the saved file in the temp directory
    """
    tmp_dir = tempfile.gettempdir()
    
    # Get file extension from original file path
    file_ext = file_path.lower().split('.')[-1] if '.' in file_path else 'csv'
    
    # Ensure extension is valid (csv, xlsx, xls, tsv)
    if file_ext not in ['csv', 'xlsx', 'xls', 'tsv']:
        file_ext = 'csv'  # Default to csv if extension is unknown
    
    final_path = os.path.join(tmp_dir, f"{user_id}_{file_id}.{file_ext}")
    
    # Write to a unique temp file first
    fd, tmp_path = tempfile.mkstemp(prefix=f"{user_id}_{file_id}_", suffix=f".{file_ext}", dir=tmp_dir)
    
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(file_bytes)
        
        # Atomic replace
        os.replace(tmp_path, final_path)
        logger.info(f"Saved file to temp directory: {final_path}")
        return final_path
    except Exception as e:
        # Clean up temp file if something goes wrong
        try:
            os.unlink(tmp_path)
        except:
            pass
        raise ValueError(f"Failed to save file to temp directory: {str(e)}")

def run_fixed_duckdb_query(local_csv_path: str, limit: int = 25) -> pd.DataFrame:
    """
    Run a fixed DuckDB query against a CSV file.
    
    Args:
        local_csv_path: Path to the CSV file in the temp directory
        limit: Maximum number of rows to return (default: 25)
    
    Returns:
        pandas DataFrame with the query results
    """
    try:
        con = duckdb.connect(database=":memory:")
        df = con.execute(
            "SELECT * FROM read_csv_auto(?, header=True) LIMIT ?;",
            [local_csv_path, int(limit)]
        ).df()
        con.close()
        logger.info(f"DuckDB query executed successfully: {len(df)} rows returned")
        return df
    except Exception as e:
        logger.error(f"Error running DuckDB query: {str(e)}")
        raise ValueError(f"Failed to run DuckDB query: {str(e)}")

# DEFAULT_TTL_DAYS = 30

# @app.post("/api/table/analyze_async")
# @require_solari_key
# def table_analyze_async():
#     """
#     Enqueue a table analysis job. Client uploads file to Storage first, writes the source doc,
#     then calls this with the source document_id to analyze.

#     Body:
#     {
#       "user_id": "...",
#       "agent_id": "...",
#       "document_id": "..."  # this is the source doc id under teams/{team}/agents/{agent}/sources/{document_id}
#     }
#     """
#     body = request.get_json(force=True) or {}

#     user_id = body.get("user_id")
#     agent_id = body.get("agent_id") or body.get("agentId")
#     document_id = body.get("document_id") or body.get("documentId")

#     if not user_id:
#         return jsonify({"status": "failure", "error": "user_id is required"}), 400
#     if not agent_id:
#         return jsonify({"status": "failure", "error": "agent_id is required"}), 400
#     if not document_id:
#         return jsonify({"status": "failure", "error": "document_id is required"}), 400

#     db = firestore.client()
#     team_id = get_team_id_for_uid(db, user_id)

#     now = utcnow()
#     expires_at = now + timedelta(days=DEFAULT_TTL_DAYS)

#     job_id = uuid.uuid4().hex
#     job_ref = (
#         db.collection("teams").document(team_id)
#           .collection("upload_jobs").document(job_id)
#     )

#     # Minimal source entry for the worker
#     source_key = f"table:{document_id}"
#     sources = [{
#         "source_key": source_key,
#         "type": "table",
#         "id": document_id,          # the agent source doc id
#         "title": document_id,       # optional; UI can replace with nickname/name from the source doc
#         "status": "queued",
#         "stage": "queued",
#         "checkpoint": {},
#         "error": None,
#         "updated_at": now,
#     }]

#     job_ref.set({
#         "job_type": "ingest_sources",
#         "connector": "table",
#         "status": "queued",
#         "created_at": now,
#         "updated_at": now,
#         "expires_at": expires_at,

#         "locked_by": None,
#         "locked_until": None,
#         "progress": 0,
#         "message": "Queued",
#         "created_by_user_id": user_id,

#         "team_id": team_id,
#         "agent_id": agent_id,

#         "sources": sources,
#     })

#     return jsonify({
#         "status": "success",
#         "job_id": job_id,
#         "team_id": team_id,
#         "queued_sources": 1,
#     }), 200

# @app.route('/api/table/analyze', methods=['POST'])
# @require_solari_key
# def analyze_table():
#     """
#     Analyze a tabular data file from a Firestore document.
    
#     Expected request body:
#     {
#         "team_id": "team123",
#         "user_id": "jyh2RyS8Mvb9OCWF7pKRKEAGxZP2",
#         "agent_id": "agent123",
#         "document_id": "miRtr9IDqCzu66rBksTG"
#     }
    
#     Process:
#     1. Get document from Firestore at teams/{team_id}/agents/{agent_id}/sources/{document_id}
#     2. Extract filePath from document
#     3. Download file from Firebase Storage
#     4. Analyze file to get row count and column types
#     5. Return results
    
#     Returns:
#         JSON response with row count and column information
#     """
#     try:
#         # Get request data
#         data = request.get_json()
        
#         if not data:
#             return jsonify({
#                 'status': 'error',
#                 'message': 'Request body is required'
#             }), 400
        
#         # Validate required fields
#         team_id = data.get('team_id') or data.get('teamId')
#         if not team_id:
#             return jsonify({
#                 'status': 'error',
#                 'message': 'team_id parameter is required'
#             }), 400

#         user_id = data.get('user_id')
#         if not user_id:
#             return jsonify({
#                 'status': 'error',
#                 'message': 'user_id parameter is required'
#             }), 400
        
#         agent_id = data.get('agent_id')
#         if not agent_id:
#             return jsonify({
#                 'status': 'error',
#                 'message': 'agent_id parameter is required'
#             }), 400
        
#         document_id = data.get('document_id')
#         if not document_id:
#             return jsonify({
#                 'status': 'error',
#                 'message': 'document_id parameter is required'
#             }), 400
        
#         logger.info(f"Analyzing table for team: {team_id}, user: {user_id}, agent: {agent_id}, document: {document_id}")
        
#         # Step 1: Get document from Firestore
#         db = firestore.client()
#         doc_ref = (
#             db.collection('teams').document(team_id)
#               .collection('agents').document(agent_id)
#               .collection('sources').document(document_id)
#         )
#         doc = doc_ref.get()
        
#         if not doc.exists:
#             return jsonify({
#                 'status': 'error',
#                 'message': f'Document not found: {document_id}'
#             }), 404
        
#         doc_data = doc.to_dict()
        
#         # Step 2: Extract filePath
#         file_path = doc_data.get('filePath')
#         if not file_path:
#             return jsonify({
#                 'status': 'error',
#                 'message': 'filePath not found in document'
#             }), 400
        
#         logger.info(f"Found filePath: {file_path}")
        
#         # Step 3: Download file from Firebase Storage
#         logger.info("Downloading file from Firebase Storage...")
#         file_content = download_file_from_firebase(file_path)
        
#         # Step 4: Analyze tabular data
#         logger.info("Analyzing tabular data...")
#         analysis_result = analyze_tabular_data(file_content, file_path)
        
#         logger.info(f"Analysis complete: {analysis_result['row_count']} rows, {len(analysis_result['columns'])} columns")
        
#         # Step 5: Update Firestore document with metadata
#         logger.info("Updating Firestore document with metadata...")
#         doc_ref.update({
#             'row_count': analysis_result['row_count'],
#             'column_count': len(analysis_result['columns'])
#         })
#         logger.info("Successfully updated document with row_count and column_count")
        
#         return jsonify({
#             'status': 'success',
#             'message': 'Successfully analyzed tabular data',
#             'document_id': document_id,
#             'file_path': file_path,
#             'row_count': analysis_result['row_count'],
#             'columns': analysis_result['columns'],
#             'column_count': len(analysis_result['columns'])
#         }), 200
        
#     except ValueError as e:
#         logger.error(f"Validation error: {str(e)}")
#         return jsonify({
#             'status': 'error',
#             'message': str(e)
#         }), 400
#     except Exception as e:
#         logger.error(f"Error analyzing table: {str(e)}", exc_info=True)
#         return jsonify({
#             'status': 'error',
#             'message': f'Failed to analyze table: {str(e)}'
#         }), 500

# @app.route('/api/table/download', methods=['POST'])
# @require_solari_key
# def download_table():
#     """
#     Test endpoint to download a table file from Firebase Storage using a Firestore source document.
    
#     Expected request body:
#     {
#         "team_id": "team123",
#         "user_id": "jyh2RyS8Mvb9OCWF7pKRKEAGxZP2",
#         "agent_id": "agent123",
#         "document_id": "MV9pGL1YP6iLdCeBxKay"
#     }
    
#     Returns:
#         JSON response with file metadata (row count, columns, types) and file size.
#         Note: file_content is bytes and not included in JSON response, but file_size is provided.
#     """
#     try:
#         # Get request data
#         data = request.get_json()
        
#         if not data:
#             return jsonify({
#                 'status': 'error',
#                 'message': 'Request body is required'
#             }), 400
        
#         # Validate required fields
#         team_id = data.get('team_id') or data.get('teamId')
#         if not team_id:
#             return jsonify({
#                 'status': 'error',
#                 'message': 'team_id parameter is required'
#             }), 400

#         user_id = data.get('user_id')
#         if not user_id:
#             return jsonify({
#                 'status': 'error',
#                 'message': 'user_id parameter is required'
#             }), 400
        
#         agent_id = data.get('agent_id')
#         if not agent_id:
#             return jsonify({
#                 'status': 'error',
#                 'message': 'agent_id parameter is required'
#             }), 400
        
#         document_id = data.get('document_id')
#         if not document_id:
#             return jsonify({
#                 'status': 'error',
#                 'message': 'document_id parameter is required'
#             }), 400
        
#         logger.info(f"Testing download_table_from_source for team: {team_id}, user: {user_id}, agent: {agent_id}, document: {document_id}")
        
#         # Call the function
#         result = download_table_from_source(team_id, user_id, agent_id, document_id)
        
#         # Prepare response (exclude file_content bytes, but include file size)
#         response_data = {
#             'status': 'success',
#             'message': 'Successfully downloaded and analyzed table',
#             'document_id': document_id,
#             'file_path': result['file_path'],
#             'file_size_bytes': len(result['file_content']),
#             'row_count': result['row_count'],
#             'columns': result['columns'],
#             'column_count': result['column_count']
#         }
        
#         logger.info(f"Test successful: {result['row_count']} rows, {result['column_count']} columns, {len(result['file_content'])} bytes")
        
#         return jsonify(response_data), 200
        
#     except ValueError as e:
#         logger.error(f"Validation error: {str(e)}")
#         return jsonify({
#             'status': 'error',
#             'message': str(e)
#         }), 400
#     except Exception as e:
#         logger.error(f"Error downloading table: {str(e)}", exc_info=True)
#         return jsonify({
#             'status': 'error',
#             'message': f'Failed to download table: {str(e)}'
#         }), 500

TABLE_LEASE_SECONDS = 120  # keep short; endpoint is doing the work synchronously

@app.route("/api/table/analyze", methods=["POST"])
@require_solari_key
def analyze_table():
    """
    Creates an upload job AND runs analysis immediately.
    Returns analysis metadata to the frontend.
    """
    try:
        data = request.get_json(silent=True) or {}

        team_id = data.get("team_id") or data.get("teamId")
        user_id = data.get("user_id") or data.get("userId")
        agent_id = data.get("agent_id") or data.get("agentId")
        document_id = data.get("document_id") or data.get("documentId")

        if not user_id:
            return jsonify({"status": "error", "message": "user_id parameter is required"}), 400
        if not agent_id:
            return jsonify({"status": "error", "message": "agent_id parameter is required"}), 400
        if not document_id:
            return jsonify({"status": "error", "message": "document_id parameter is required"}), 400

        db = firestore.client()

        # Optional: allow team_id omitted
        if not team_id:
            try:
                team_id = get_team_id_for_uid(db, user_id)
            except Exception:
                return jsonify({"status": "error", "message": "team_id parameter is required"}), 400

        # Load agent source doc to get filePath
        doc_ref = (
            db.collection("teams").document(team_id)
              .collection("agents").document(agent_id)
              .collection("sources").document(document_id)
        )
        snap = doc_ref.get()
        if not snap.exists:
            return jsonify({"status": "error", "message": f"Document not found: {document_id}"}), 404

        doc_data = snap.to_dict() or {}
        file_path = doc_data.get("filePath")
        if not file_path:
            return jsonify({"status": "error", "message": "filePath not found in document"}), 400

        now_unix = int(time.time())

        # --- Create an upload job doc under teams/{teamId}/upload_jobs/{jobId}
        job_id = f"table_{now_unix}_{uuid.uuid4().hex[:8]}"
        job_ref = (
            db.collection("teams").document(team_id)
              .collection("upload_jobs").document(job_id)
        )

        source_key = f"table:{document_id}"

        # Mark analysis as processing on the agent source doc (canonical)
        doc_ref.update({
            "analysis_status": "processing",
            "analysis_message": "Analyzing table",
            "analysis_job_id": job_id,
            "updated_at_unix": now_unix,
        })

        # Create job in "processing" so worker doesn't double-run while the API is running
        job_ref.set({
            "job_type": "analyze_table",
            "connector": "table",
            "team_id": team_id,
            "agent_id": agent_id,
            "created_by_user_id": user_id,

            "status": "processing",
            "message": "Processing",
            "progress": 0,

            "created_at_unix": now_unix,
            "updated_at_unix": now_unix,
            "expires_at_unix": now_unix + 30 * 24 * 3600,

            "locked_by": "api",
            "locked_until": utcnow() + timedelta(seconds=TABLE_LEASE_SECONDS),

            "sources": [{
                "source_key": source_key,
                "type": "table",
                "id": document_id,
                "title": doc_data.get("nickname") or doc_data.get("name") or document_id,
                "file_path": file_path,
                "status": "processing",
                "stage": "analyze",
                "error": None,
                "result": None,
                "checkpoint": {},
            }]
        })

        # --- Do the analysis now (synchronous)
        logger.info(f"[table/analyze] downloading file: {file_path}")
        file_content = download_file_from_firebase(file_path)

        logger.info("[table/analyze] analyzing tabular data")
        analysis_result = analyze_tabular_data(file_content, file_path)

        row_count = int(analysis_result.get("row_count") or 0)
        columns = analysis_result.get("columns") or {}
        column_count = int(len(columns))

        # --- Write canonical results on agent source doc
        doc_ref.update({
            "row_count": row_count,
            "columns": columns,
            "column_count": column_count,
            "analysis_status": "done",
            "analysis_message": "Table analyzed successfully",
            "analyzed_at_unix": now_unix,
            "updated_at_unix": now_unix,
        })

        # --- Mark job done
        update_source(job_ref, source_key, {
            "status": "done",
            "stage": "done",
            "error": None,
            "result": {
                "row_count": row_count,
                "column_count": column_count,
            }
        })
        now_unix = int(utcnow().timestamp())

        update_job(job_ref, status="done", progress=100, message="Completed")
        update_job(job_ref, locked_by=None, locked_until=None,)

        # --- Return what frontend expects
        return jsonify({
            "status": "success",
            "message": "Table analyzed successfully",
            "document_id": document_id,
            "file_path": file_path,
            "row_count": row_count,
            "columns": columns,
            "column_count": column_count,
            "job_id": job_id,
        }), 200

    except Exception as e:
        logger.error(f"Error analyzing table: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": f"Failed to analyze table: {str(e)}"
        }), 500

@app.route('/api/table/prepare', methods=['POST'])
@require_solari_key
def prepare_table():
    """
    Download table, extract metadata, and save to temp folder for DuckDB processing.
    
    Expected request body:
    {
        "team_id": "team123",
        "user_id": "jyh2RyS8Mvb9OCWF7pKRKEAGxZP2",
        "agent_id": "agent123",
        "document_id": "MV9pGL1YP6iLdCeBxKay"
    }
    
    Process:
    1. Download file from Firebase Storage
    2. Extract metadata (row count, columns, types)
    3. Save file to temp directory atomically
    4. Return metadata and temp file path
    
    Returns:
        JSON response with metadata and temp file path ready for DuckDB
    """
    try:
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'Request body is required'
            }), 400
        
        # Validate required fields
        team_id = data.get('team_id') or data.get('teamId')
        if not team_id:
            return jsonify({
                'status': 'error',
                'message': 'team_id parameter is required'
            }), 400

        user_id = data.get('user_id')
        if not user_id:
            return jsonify({
                'status': 'error',
                'message': 'user_id parameter is required'
            }), 400
        
        agent_id = data.get('agent_id')
        if not agent_id:
            return jsonify({
                'status': 'error',
                'message': 'agent_id parameter is required'
            }), 400
        
        document_id = data.get('document_id')
        if not document_id:
            return jsonify({
                'status': 'error',
                'message': 'document_id parameter is required'
            }), 400
        
        logger.info(f"Preparing table for DuckDB: team: {team_id}, user: {user_id}, agent: {agent_id}, document: {document_id}")
        
        # Step 1: Download file and get metadata
        logger.info("Step 1: Downloading table and extracting metadata...")
        download_result = download_table_from_source(team_id, user_id, agent_id, document_id)
        
        # Step 2: Save file to temp directory
        logger.info("Step 2: Saving file to temp directory...")
        temp_file_path = materialize_table_file_atomic(
            file_bytes=download_result['file_content'],
            user_id=user_id,
            file_id=document_id,
            file_path=download_result['file_path']
        )
        
        logger.info(f"Successfully prepared table: {temp_file_path}")
        
        # Prepare response
        response_data = {
            'status': 'success',
            'message': 'Successfully downloaded, analyzed, and saved table to temp directory',
            'document_id': document_id,
            'file_path': download_result['file_path'],
            'temp_file_path': temp_file_path,
            'file_size_bytes': len(download_result['file_content']),
            'row_count': download_result['row_count'],
            'columns': download_result['columns'],
            'column_count': download_result['column_count']
        }
        
        return jsonify(response_data), 200
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400
    except Exception as e:
        logger.error(f"Error preparing table: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'Failed to prepare table: {str(e)}'
        }), 500

@app.route('/api/table/query', methods=['POST'])
@require_solari_key
def query_table():
    """
    Download table, save to temp folder, run DuckDB query, and return results.
    
    Expected request body:
    {
        "team_id": "team123",
        "user_id": "jyh2RyS8Mvb9OCWF7pKRKEAGxZP2",
        "agent_id": "agent123",
        "document_id": "MV9pGL1YP6iLdCeBxKay",
        "limit": 25  # optional, defaults to 25
    }
    
    Process:
    1. Download file from Firebase Storage and get metadata
    2. Save file to temp directory (as CSV)
    3. Run fixed DuckDB query against the CSV
    4. Return query results as JSON
    
    Returns:
        JSON response with query results (columns, rows, rows_returned)
    """
    try:
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'Request body is required'
            }), 400
        
        # Validate required fields
        team_id = data.get('team_id') or data.get('teamId')
        if not team_id:
            return jsonify({
                'success': False,
                'error': 'team_id parameter is required'
            }), 400

        user_id = data.get('user_id')
        if not user_id:
            return jsonify({
                'success': False,
                'error': 'user_id parameter is required'
            }), 400
        
        agent_id = data.get('agent_id')
        if not agent_id:
            return jsonify({
                'success': False,
                'error': 'agent_id parameter is required'
            }), 400
        
        document_id = data.get('document_id')
        if not document_id:
            return jsonify({
                'success': False,
                'error': 'document_id parameter is required'
            }), 400
        
        # Optional parameter
        limit = data.get('limit', 25)
        if not isinstance(limit, int) or limit < 1:
            limit = 25
        
        logger.info(f"Querying table: team: {team_id}, user: {user_id}, agent: {agent_id}, document: {document_id}, limit: {limit}")
        
        # Step 1: Download file and get metadata
        logger.info("Step 1: Downloading table and extracting metadata...")
        download_result = download_table_from_source(team_id, user_id, agent_id, document_id)
        
        # Step 2: Save file to temp directory (force .csv extension since files are converted to CSV on upload)
        logger.info("Step 2: Saving file to temp directory as CSV...")
        # Create a file_path with .csv extension for the materialize function
        csv_file_path = download_result['file_path']
        if not csv_file_path.lower().endswith('.csv'):
            csv_file_path = csv_file_path.rsplit('.', 1)[0] + '.csv'
        
        temp_file_path = materialize_table_file_atomic(
            file_bytes=download_result['file_content'],
            user_id=user_id,
            file_id=document_id,
            file_path=csv_file_path
        )
        
        # Step 3: Run DuckDB query
        logger.info("Step 3: Running DuckDB query...")
        df_out = run_fixed_duckdb_query(temp_file_path, limit=limit)
        
        # Step 4: Prepare response
        logger.info(f"Query successful: {len(df_out)} rows returned")
        
        return jsonify({
            'success': True,
            'table': {
                'columns': list(df_out.columns),
                'rows': df_out.to_dict(orient='records'),
                'rows_returned': len(df_out)
            }
        }), 200
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400
    except Exception as e:
        logger.error(f"Error querying table: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'Failed to query table: {str(e)}'
        }), 500

ALLOWED_KEYS = {"select", "filters", "groupby", "metrics", "sort", "limit", "assumptions"}
CANON_TYPES = {"number", "date", "category", "text", "boolean"}
ALLOWED_FILTER_OPS = {"==", "!=", ">", "<", ">=", "<=", "contains"}
ALLOWED_AGGS = {"sum", "avg", "min", "max", "count"}
DEFAULT_LIMIT=50
MAX_LIMIT = 200


def build_planner_schema(column_metadata: Dict[str, Any]) -> list[dict]:
    schema = []
    for col, meta in column_metadata.items():
        schema.append({
            "name": col,
            "canon_type": meta.get("canon_type", "text"),
            "examples": (meta.get("examples") or [])[:3],
        })
    return schema

def extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    m = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if not m:
        raise ValueError(f"Planner did not return JSON. Got: {text[:200]}")
    return json.loads(m.group(1))

def query_to_plan(question: str, column_metadata: Dict[str, Any], llm_call) -> Dict[str, Any]:
    planner_schema = build_planner_schema(column_metadata)

    system = """You are a query planner for a single table.
Return ONLY a valid JSON object and nothing else. Do NOT output SQL.

You MUST output EXACTLY these top-level keys (no more, no less):
select, filters, groupby, metrics, sort, limit, assumptions

CRITICAL FORMAT RULES:
1) select MUST be a list of BARE column names only.
   - No functions, no parentheses, no aliases, no expressions.
   - Example OK: ["month"]
   - Example NOT OK: ["sum(revenue_usd) as total_sales"]

2) metrics MUST be a list of objects in this exact shape:
   {"agg": "<agg>", "col": "<column_name or null>", "as": "<alias>"}
   - agg must be one of: sum, avg, min, max, count
   - For agg="count", col may be null (or omitted only if you set it explicitly to null).
   - For all other aggs, col is REQUIRED and must be a number column.
   - Do NOT use shorthand like {"total_sales":"sum"}.

3) sort MUST be a list of objects in this exact shape:
   {"col": "<column or metric alias>", "dir": "asc"|"desc"}
   - Do NOT use shorthand like {"total_sales":"desc"}.

4) filters MUST be a list of objects in this exact shape:
   {"col":"<column>", "op":"<op>", "value": <literal>}
   Allowed ops: ==, !=, >, <, >=, <=, contains

5) limit MUST be an integer between 1 and 200. Default to 200 if not specified.

TYPE RULES by canon_type:
- number: supports > < >= <= and aggs sum/avg/min/max/count
- date: supports > < >= <= and aggs min/max/count; groupby allowed
- category: supports == != and groupby allowed
- text: supports contains, == != (avoid groupby unless explicitly asked)
- boolean: supports == != with true/false; count allowed

COLUMN RULE:
- Only use columns that exist in the provided schema.
- Never invent column names.

If ambiguous, make your best guess and explain in assumptions (string).
If no assumptions, assumptions must be "".
"""
    user = f"""Table schema (JSON):
    {json.dumps(planner_schema, ensure_ascii=False)}

    Question:
    {question}

    Remember: select = bare columns only. metrics must be objects with agg/col/as. Return JSON only.
    """

    raw = llm_call(system=system, user=user)  # <-- you implement this
    plan = extract_json_object(raw)

    # keep only the keys you allow
    plan = {k: plan[k] for k in plan if k in ALLOWED_KEYS}

    # defaults so downstream code never crashes
    plan.setdefault("select", [])
    plan.setdefault("filters", [])
    plan.setdefault("groupby", [])
    plan.setdefault("metrics", [])
    plan.setdefault("sort", [])
    plan.setdefault("limit", 200)
    plan.setdefault("assumptions", "")

    return plan

def get_table_meta(team_id: str, user_id: str, document_id: str, agent_id: str = None) -> Dict[str, Any]:
    """
    Get table metadata from Firestore source document.
    
    Args:
        team_id: Team identifier
        user_id: User identifier
        document_id: Source document identifier (file_id)
        agent_id: Agent identifier (optional, defaults to 'default' if not provided)
    
    Returns:
        Dictionary with columnMetadata in the format expected by build_planner_schema
    """
    db = firestore.client()
    
    # If agent_id not provided, use 'default' as fallback
    if not agent_id:
        agent_id = 'default'
        logger.warning(f"agent_id not provided, using 'default'")
    
    doc_ref = (
        db.collection('teams').document(team_id)
          .collection('agents').document(agent_id)
          .collection('sources').document(document_id)
    )
    doc = doc_ref.get()
    
    if not doc.exists:
        raise ValueError(f'Document not found: {document_id}')
    
    doc_data = doc.to_dict()
    
    # Get columnMetadata from document (as stored in Firestore)
    column_metadata_raw = doc_data.get('columnMetadata', {})
    
    if not column_metadata_raw:
        raise ValueError(f'No columnMetadata found in document. Please analyze the table first using /api/table/analyze')
    
    # Map from Firestore format to planner format
    # Firestore has: canon_dtype, example_values
    # Planner expects: canon_type, examples
    column_metadata = {}
    
    for col_name, col_info in column_metadata_raw.items():
        # Map canon_dtype -> canon_type (handle both field names for compatibility)
        canon_type = col_info.get('canon_type') or col_info.get('canon_dtype', 'text')
        
        # Map example_values -> examples (handle both field names)
        examples = col_info.get('examples') or col_info.get('example_values', [])
        
        column_metadata[col_name] = {
            'canon_type': canon_type,
            'examples': examples
        }
    
    return {
        'columnMetadata': column_metadata
    }

def planner_llm_call(system: str, user: str) -> str:
    """
    Wrapper function for LLM calls that matches the signature expected by query_to_plan.
    
    Args:
        system: System prompt
        user: User prompt
    
    Returns:
        LLM response text
    """
    openai_client = get_openai_client()
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content

@app.route("/api/table/plan", methods=["POST"])
@require_solari_key
def table_plan():
    """
    Full pipeline: Generate a SQL plan from a question, validate, and fix if needed.
    
    Expected request body:
    {
      "team_id": "...",
      "user_id": "...",
      "source_id": "...",  # This is the document_id
      "question": "...",
      "agent_id": "..."  # Optional, defaults to 'default'
    }
    
    Returns:
        JSON response with raw_plan, normalized_plan, final_plan, fixes_applied, issues, suggestions, planner_schema
    """
    try:
        body = request.get_json(force=True)
        
        if not body:
            return jsonify({
                'success': False,
                'error': 'Request body is required'
            }), 400
        
        team_id = body.get("team_id") or body.get("teamId")
        if not team_id:
            return jsonify({
                'success': False,
                'error': 'team_id is required'
            }), 400

        user_id = body.get("user_id")
        if not user_id:
            return jsonify({
                'success': False,
                'error': 'user_id is required'
            }), 400
        
        source_id = body.get("source_id")
        if not source_id:
            return jsonify({
                'success': False,
                'error': 'source_id is required'
            }), 400
        
        question = body.get("question")
        if not question:
            return jsonify({
                'success': False,
                'error': 'question is required'
            }), 400
        
        agent_id = body.get("agent_id")  # Optional
        
        # Step 1: Get table metadata
        logger.info(f"Getting table metadata for team: {team_id}, user: {user_id}, agent_id: {agent_id}, source_id: {source_id}")
        table_meta = get_table_meta(team_id, user_id, source_id, agent_id=agent_id)
        column_metadata = table_meta["columnMetadata"]
        
        if not column_metadata:
            return jsonify({
                'success': False,
                'error': 'No column metadata found. Please analyze the table first using /api/table/analyze'
            }), 400
        
        # Step 2: Generate raw plan from LLM
        logger.info(f"Generating plan for question: {question[:100]}...")
        raw_plan = query_to_plan(question, column_metadata, llm_call=planner_llm_call)
        
        # Step 3: Normalize the plan
        normalized_plan = normalize_plan_minimal(raw_plan)
        
        # Step 4: Build planner schema and get column types
        planner_schema = build_planner_schema_from_table_meta(table_meta, max_examples=3)
        col_types = col_types_from_planner_schema(planner_schema)
        
        # Step 5: Validate and fix if needed
        fix_result = validate_then_fix_once(normalized_plan, col_types)
        
        # Build response
        response = {
            "success": fix_result["success"],
            "raw_plan": raw_plan,
            "normalized_plan": normalized_plan,
            "final_plan": fix_result["plan"],
            "fixes_applied": fix_result.get("fixes_applied", []),
            "issues": fix_result.get("issues", []),
            "suggestions": fix_result.get("suggestions", []),
            "planner_schema": planner_schema
        }
        
        return jsonify(response), 200
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400
    except Exception as e:
        logger.error(f"Error generating plan: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route("/api/table/plan/fix", methods=["POST"])
@require_solari_key
def fix_table_plan():
    """
    Fix a plan by validating it and applying fuzzy column name corrections.
    
    Expected request body:
    {
      "team_id": "...",
      "user_id": "...",
      "source_id": "...",  # This is the document_id
      "plan": {...},
      "agent_id": "..."  # Optional, defaults to 'default'
    }
    
    Returns:
        JSON response with normalized_plan, final_plan, fixes_applied, issues, suggestions, planner_schema
    """
    try:
        body = request.get_json(force=True)
        
        if not body:
            return jsonify({
                'success': False,
                'error': 'Request body is required'
            }), 400
        
        team_id = body.get("team_id") or body.get("teamId")
        if not team_id:
            return jsonify({
                'success': False,
                'error': 'team_id is required'
            }), 400

        user_id = body.get("user_id")
        if not user_id:
            return jsonify({
                'success': False,
                'error': 'user_id is required'
            }), 400
        
        source_id = body.get("source_id")
        if not source_id:
            return jsonify({
                'success': False,
                'error': 'source_id is required'
            }), 400
        
        raw_plan = body.get("plan")
        if not raw_plan:
            return jsonify({
                'success': False,
                'error': 'plan is required'
            }), 400
        
        agent_id = body.get("agent_id")  # Optional
        
        # Step 1: Get table metadata
        logger.info(f"Getting table metadata for team: {team_id}, user: {user_id}, agent_id: {agent_id}, source_id: {source_id}")
        table_meta = get_table_meta(team_id, user_id, source_id, agent_id=agent_id)
        column_metadata = table_meta["columnMetadata"]
        
        if not column_metadata:
            return jsonify({
                'success': False,
                'error': 'No column metadata found. Please analyze the table first using /api/table/analyze'
            }), 400
        
        # Step 2: Filter plan to only allowed keys (keep original for raw_plan)
        filtered_plan = {k: raw_plan[k] for k in raw_plan if k in ALLOWED_KEYS}
        
        # Step 3: Normalize the plan
        normalized_plan = normalize_plan_minimal(filtered_plan)
        
        # Step 4: Build planner schema and get column types
        planner_schema = build_planner_schema_from_table_meta(table_meta, max_examples=3)
        col_types = col_types_from_planner_schema(planner_schema)
        
        # Step 5: Validate and fix if needed
        fix_result = validate_then_fix_once(normalized_plan, col_types)
        
        # Build response
        response = {
            "success": fix_result["success"],
            "raw_plan": raw_plan,  # Input plan
            "normalized_plan": normalized_plan,
            "final_plan": fix_result["plan"],
            "fixes_applied": fix_result.get("fixes_applied", []),
            "issues": fix_result.get("issues", []),
            "suggestions": fix_result.get("suggestions", []),
            "planner_schema": planner_schema
        }
        
        return jsonify(response), 200
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400
    except Exception as e:
        logger.error(f"Error fixing plan: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

def build_planner_schema_from_table_meta(table_meta: Dict[str, Any], max_examples: int = 3) -> List[Dict[str, Any]]:
    """
    table_meta["columnMetadata"] is expected to be a map:
      { "month": {"canon_type":"category", "examples":[...]}, ... }
    """
    column_metadata = table_meta.get("columnMetadata") or {}
    out: List[Dict[str, Any]] = []
    for col, meta in column_metadata.items():
        canon = str(meta.get("canon_type", "text")).lower()
        if canon not in CANON_TYPES:
            canon = "text"
        examples = meta.get("examples") or []
        out.append({"name": col, "canon_type": canon, "examples": examples[:max_examples]})
    return out

def col_types_from_planner_schema(planner_schema: List[Dict[str, Any]]) -> Dict[str, str]:
    m: Dict[str, str] = {}
    for c in planner_schema:
        name = str(c.get("name", "")).strip()
        if not name:
            continue
        t = str(c.get("canon_type", "text")).lower()
        if t not in CANON_TYPES:
            t = "text"
        m[name] = t
    return m

def looks_like_sql(expr: str) -> bool:
    s = expr.lower()
    if "(" in s or ")" in s:
        return True
    if " as " in s:
        return True
    for tok in ["select ", " from ", " where ", "group by", "order by", "limit "]:
        if tok in s:
            return True
    return False

def clamp_limit(x: Any) -> int:
    try:
        v = int(x)
    except Exception:
        v = DEFAULT_LIMIT
    return max(1, min(MAX_LIMIT, v))

def normalize_plan_minimal(raw_plan: Dict[str, Any]) -> Dict[str, Any]:
    """
    Minimal normalization:
    - ensure keys exist
    - clamp limit
    - normalize sort shorthand [{alias:"desc"}] -> [{"col":alias,"dir":"desc"}]
    - normalize filter op "=" -> "=="
    (does NOT fix metrics/select — just standardizes structure a bit)
    """
    p = dict(raw_plan or {})
    p.setdefault("select", [])
    p.setdefault("filters", [])
    p.setdefault("groupby", [])
    p.setdefault("metrics", [])
    p.setdefault("sort", [])
    p.setdefault("assumptions", "")
    p["limit"] = clamp_limit(p.get("limit", DEFAULT_LIMIT))

    # filters: "=" -> "=="
    nf = []
    for f in p["filters"]:
        if not isinstance(f, dict):
            continue
        op = f.get("op")
        if op == "=":
            f = dict(f)
            f["op"] = "=="
        nf.append(f)
    p["filters"] = nf

    # sort: shorthand -> canonical
    ns = []
    for s in p["sort"]:
        if isinstance(s, dict) and "col" in s and "dir" in s:
            ns.append({"col": str(s["col"]).strip(), "dir": str(s["dir"]).strip().lower()})
        elif isinstance(s, dict) and len(s) == 1:
            k, v = next(iter(s.items()))
            ns.append({"col": str(k).strip(), "dir": str(v).strip().lower()})
    p["sort"] = ns

    # coerce lists of strings
    p["select"] = [x for x in p["select"] if isinstance(x, str)]
    p["groupby"] = [x for x in p["groupby"] if isinstance(x, str)]

    return p

def sanity_check_plan(plan: Dict[str, Any], col_types: Dict[str, str]) -> Tuple[bool, List[str], List[str]]:
    cols = set(col_types.keys())
    issues: List[str] = []
    suggestions: List[str] = []

    # ---- 1) Column existence checks ----
    for c in plan.get("select", []):
        if c not in cols and looks_like_sql(c):
            issues.append(f"select contains SQL/expression '{c}'. select must contain only bare column names.")
        elif c not in cols:
            issues.append(f"Unknown select column: '{c}'")

    for c in plan.get("groupby", []):
        if c not in cols:
            issues.append(f"Unknown groupby column: '{c}'")
        else:
            if col_types[c] == "text":
                issues.append(f"groupby uses text column '{c}'. Prefer canon_type='category' or 'date'.")
                suggestions.append(f"Consider setting '{c}' canon_type to 'category' (values look like categories).")

    # ---- 2) Filters: columns + ops allowed by type ----
    for f in plan.get("filters", []):
        if not isinstance(f, dict):
            issues.append("filters contains a non-object entry.")
            continue
        col = f.get("col")
        op = f.get("op")
        if col not in cols:
            issues.append(f"Unknown filter column: '{col}'")
            continue
        if op not in ALLOWED_FILTER_OPS:
            issues.append(f"Unsupported filter op '{op}' for column '{col}'.")
            continue

        t = col_types[col]
        if op == "contains" and t != "text":
            issues.append(f"Invalid filter: op 'contains' used on {t} column '{col}'.")
        if op in {">", "<", ">=", "<="} and t not in {"number", "date"}:
            issues.append(f"Invalid filter: op '{op}' used on {t} column '{col}' (only number/date allowed).")

    # ---- 3) Metrics: must be canonical + agg allowed for type ----
    # Your sample has shorthand metrics [{"total_sales":"sum"}] which is ambiguous (missing col).
    for m in plan.get("metrics", []):
        if not isinstance(m, dict):
            issues.append("metrics contains a non-object entry.")
            continue

        agg = str(m.get("agg", "")).lower()
        
        # For count, col is optional. For other aggs, col is required.
        if agg == "count":
            # Count metric: only requires agg and as
            if "agg" not in m or "as" not in m:
                issues.append("Count metric requires 'agg' and 'as' fields. 'col' is optional.")
            else:
                if agg not in ALLOWED_AGGS:
                    issues.append(f"Unsupported aggregation '{agg}'.")
                alias = m.get("as")
                if not alias:
                    issues.append("Metric missing alias field 'as'.")
                # col is optional for count, but if provided, we can validate it exists
                col = m.get("col")
                if col and col not in cols:
                    issues.append(f"Unknown metric column: '{col}' (optional for count)")
        elif {"agg", "col", "as"} <= set(m.keys()):
            # Non-count metrics: require agg, col, and as
            col = m.get("col")
            alias = m.get("as")
            if agg not in ALLOWED_AGGS:
                issues.append(f"Unsupported aggregation '{agg}'.")
            if col not in cols:
                issues.append(f"Unknown metric column: '{col}'")
            elif col_types[col] != "number":
                issues.append(f"Aggregation '{agg}' requires number column, got {col_types[col]} for '{col}'.")
            if not alias:
                issues.append("Metric missing alias field 'as'.")
        elif len(m) == 1:
            alias, agg_val = next(iter(m.items()))
            issues.append(
                f"Metric shorthand detected: {{'{alias}':'{agg_val}'}} is missing the source column. "
                f"Expected {{'agg':'{agg_val}','col':'<numeric_col>','as':'{alias}'}} (or for count: {{'agg':'count','as':'{alias}'}})."
            )
            suggestions.append(
                "Update planner prompt to force metrics objects with keys: agg, col (optional for count), as (no shorthand)."
            )
        else:
            issues.append("Metric object is in an unsupported shape. Expected keys: agg, col (optional for count), as.")

    # ---- 4) Limit guardrail ----
    lim = plan.get("limit")
    if not isinstance(lim, int) or lim < 1 or lim > MAX_LIMIT:
        issues.append(f"limit must be an integer between 1 and {MAX_LIMIT}.")

    ok = len(issues) == 0
    return ok, issues, suggestions

# --------------------------
# Main wrapper: uses your get_table_meta + current LLM output
# --------------------------

def step8_sanity_check(
    team_id: str,
    user_id: str,
    document_id: str,
    agent_id: str,
    llm_output: Dict[str, Any],
    get_table_meta_fn,
) -> Dict[str, Any]:
    """
    get_table_meta_fn should be your function:
      get_table_meta(team_id, user_id, document_id, agent_id) -> dict (table_meta)

    llm_output is what you pasted:
      {"plan": {...}, "planner_schema": [...], ...}

    Returns a dict with ok/issues and debug info.
    """
    table_meta = get_table_meta_fn(team_id, user_id, document_id, agent_id)

    # Prefer schema from Firestore (source of truth), not from llm_output
    planner_schema = build_planner_schema_from_table_meta(table_meta, max_examples=3)
    col_types = col_types_from_planner_schema(planner_schema)

    raw_plan = llm_output.get("plan") or {}
    normalized_plan = normalize_plan_minimal(raw_plan)

    ok, issues, suggestions = sanity_check_plan(normalized_plan, col_types)

    return {
        "success": ok,
        "issues": issues,
        "suggestions": suggestions,
        "normalized_plan": normalized_plan,
        "planner_schema": planner_schema,  # debug
    }

def _norm_col(s: str) -> str:
    s = str(s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("-", " ").replace("_", " ")
    return s

def suggest_column(target: str, known_cols: List[str], cutoff: float = 0.86) -> Optional[str]:
    """
    Conservative fuzzy match:
    - exact match after normalization
    - difflib close match on normalized names (with high cutoff)
    Returns the best matching real column name, or None.
    """
    if not target or not known_cols:
        return None

    norm_to_real = {_norm_col(c): c for c in known_cols}
    tnorm = _norm_col(target)

    # exact after normalization
    if tnorm in norm_to_real:
        return norm_to_real[tnorm]

    # close match among normalized names
    candidates = difflib.get_close_matches(tnorm, list(norm_to_real.keys()), n=1, cutoff=cutoff)
    if candidates:
        return norm_to_real[candidates[0]]

    return None

def fix_unknown_columns(plan: Dict[str, Any], known_cols: List[str]) -> Tuple[Dict[str, Any], List[str]]:
    """
    Attempts to fix unknown column references in:
      - select
      - groupby
      - filters[].col
      - metrics[].col

    Returns (fixed_plan, fixes_applied_descriptions).
    Only applies low-risk name fixes (typos/formatting).
    """
    fixes: List[str] = []
    fixed = {**plan}
    cols_set = set(known_cols)

    # select
    new_select = []
    for c in fixed.get("select", []) or []:
        if c in cols_set:
            new_select.append(c)
            continue
        sug = suggest_column(c, known_cols)
        if sug and sug != c:
            fixes.append(f"select: '{c}' → '{sug}'")
            new_select.append(sug)
        else:
            new_select.append(c)
    fixed["select"] = new_select

    # groupby
    new_groupby = []
    for c in fixed.get("groupby", []) or []:
        if c in cols_set:
            new_groupby.append(c)
            continue
        sug = suggest_column(c, known_cols)
        if sug and sug != c:
            fixes.append(f"groupby: '{c}' → '{sug}'")
            new_groupby.append(sug)
        else:
            new_groupby.append(c)
    fixed["groupby"] = new_groupby

    # filters
    new_filters = []
    for f in fixed.get("filters", []) or []:
        if not isinstance(f, dict):
            new_filters.append(f)
            continue
        col = f.get("col")
        if col in cols_set:
            new_filters.append(f)
            continue
        sug = suggest_column(str(col), known_cols)
        if sug and sug != col:
            fixes.append(f"filter.col: '{col}' → '{sug}'")
            f2 = {**f, "col": sug}
            new_filters.append(f2)
        else:
            new_filters.append(f)
    fixed["filters"] = new_filters

    # metrics
    new_metrics = []
    for m in fixed.get("metrics", []) or []:
        if not isinstance(m, dict):
            new_metrics.append(m)
            continue

        agg = (m.get("agg") or "").lower()
        col = m.get("col")

        # don't invent metric columns; only fix if provided but misspelled
        if not col or agg == "count":
            new_metrics.append(m)
            continue

        if col in cols_set:
            new_metrics.append(m)
            continue

        sug = suggest_column(str(col), known_cols)
        if sug and sug != col:
            fixes.append(f"metric.col: '{col}' → '{sug}'")
            m2 = {**m, "col": sug}
            new_metrics.append(m2)
        else:
            new_metrics.append(m)
    fixed["metrics"] = new_metrics

    # sort (optional): only fix if you ONLY allow sorting by real columns
    # If you allow sorting by metric aliases, skip fixing sort here.
    # Uncomment if you want strict sort columns.
    #
    # new_sort = []
    # for s in fixed.get("sort", []) or []:
    #     if not isinstance(s, dict):
    #         new_sort.append(s); continue
    #     col = s.get("col")
    #     if col in cols_set:
    #         new_sort.append(s); continue
    #     sug = suggest_column(str(col), known_cols)
    #     if sug and sug != col:
    #         fixes.append(f"sort.col: '{col}' → '{sug}'")
    #         new_sort.append({**s, "col": sug})
    #     else:
    #         new_sort.append(s)
    # fixed["sort"] = new_sort

    return fixed, fixes

def validate_then_fix_once(
    normalized_plan: Dict[str, Any],
    col_types: Dict[str, str],
) -> Dict[str, Any]:
    """
    Validates the plan using sanity_check_plan, and if there are unknown-column issues,
    attempts to fix them using fuzzy matching, then revalidates once.
    
    Args:
        normalized_plan: The normalized plan to validate
        col_types: Dictionary mapping column names to their canon_types
    
    Returns:
        Dictionary with success, plan, issues, suggestions, and fixes_applied
    """
    # Create a wrapper function that calls sanity_check_plan with col_types
    def validate_fn(plan):
        return sanity_check_plan(plan, col_types)
    
    # Get known columns from col_types keys
    known_cols = list(col_types.keys())
    
    ok, issues, suggestions = validate_fn(normalized_plan)

    if ok:
        return {
            "success": True,
            "plan": normalized_plan,
            "issues": [],
            "suggestions": suggestions,
            "fixes_applied": []
        }

    # Only attempt name fixes if there are unknown-column issues
    has_unknown_col_issue = any("Unknown" in msg and "column" in msg for msg in issues)
    if not has_unknown_col_issue:
        return {
            "success": False,
            "plan": normalized_plan,
            "issues": issues,
            "suggestions": suggestions,
            "fixes_applied": []
        }

    fixed_plan, fixes_applied = fix_unknown_columns(normalized_plan, known_cols)

    ok2, issues2, suggestions2 = validate_fn(fixed_plan)
    return {
        "success": ok2,
        "plan": fixed_plan if ok2 else normalized_plan,
        "issues": [] if ok2 else issues2,
        "suggestions": suggestions2,
        "fixes_applied": fixes_applied
    }

def validate_then_fix_once_with_sanity(normalized_plan: dict, col_types: dict) -> dict:
    known_cols = list(col_types.keys())

    ok, issues, suggestions = sanity_check_plan(normalized_plan, col_types)
    if ok:
        return {
            "success": True,
            "final_plan": normalized_plan,
            "issues": [],
            "suggestions": suggestions,
            "fixes_applied": [],
        }

    has_unknown_col_issue = any("Unknown" in msg and "column" in msg for msg in issues)
    if not has_unknown_col_issue:
        return {
            "success": False,
            "final_plan": normalized_plan,
            "issues": issues,
            "suggestions": suggestions,
            "fixes_applied": [],
        }

    fixed_plan, fixes_applied = fix_unknown_columns(normalized_plan, known_cols)
    ok2, issues2, suggestions2 = sanity_check_plan(fixed_plan, col_types)
    return {
        "success": ok2,
        "final_plan": fixed_plan if ok2 else normalized_plan,
        "issues": [] if ok2 else issues2,
        "suggestions": suggestions2,
        "fixes_applied": fixes_applied,
    }

def _filter_allowed_plan_keys(raw_plan: dict) -> dict:
    raw_plan = raw_plan or {}
    return {k: raw_plan[k] for k in raw_plan if k in ALLOWED_KEYS}

def download_table_csv_to_temp(team_id: str, user_id: str, source_id: str, agent_id: str) -> str:
    """
    Download table file and save to temp directory as CSV.
    Returns the local file path.
    Reuses download_table_from_source and materialize_table_file_atomic.
    """
    download_result = download_table_from_source(team_id, user_id, agent_id, source_id)
    
    # Ensure CSV extension
    csv_file_path = download_result['file_path']
    if not csv_file_path.lower().endswith('.csv'):
        csv_file_path = csv_file_path.rsplit('.', 1)[0] + '.csv'
    
    temp_file_path = materialize_table_file_atomic(
        file_bytes=download_result['file_content'],
        user_id=user_id,
        file_id=source_id,
        file_path=csv_file_path
    )
    
    return temp_file_path

def plan_to_sql(plan: Dict[str, Any]) -> str:
    """
    Convert a plan dictionary to SQL query for DuckDB.
    """
    # Build SELECT clause
    select_parts = []
    
    # Get columns that are being aggregated in metrics
    # When there's a GROUP BY, we shouldn't include these columns as raw columns in SELECT
    groupby_cols = plan.get("groupby", [])
    aggregated_cols = set()
    for metric in plan.get("metrics", []):
        col = metric.get("col")
        if col:
            aggregated_cols.add(col)
    
    # Add select columns (but exclude columns that are being aggregated if there's a GROUP BY)
    for col in plan.get("select", []):
        # If there's a GROUP BY and this column is being aggregated, skip it
        # The aggregate will provide the value, so we don't need the raw column
        if groupby_cols and col in aggregated_cols:
            continue
        select_parts.append(f'"{col}"')
    
    # Add metrics/aggregations
    for metric in plan.get("metrics", []):
        agg = metric.get("agg", "").lower()
        col = metric.get("col")
        alias = metric.get("as")
        
        if agg == "count":
            if col:
                select_parts.append(f'COUNT("{col}") AS "{alias}"')
            else:
                select_parts.append(f'COUNT(*) AS "{alias}"')
        elif col:
            select_parts.append(f'{agg.upper()}("{col}") AS "{alias}"')
    
    if not select_parts:
        select_parts = ["*"]
    
    select_clause = ", ".join(select_parts)
    
    # Build FROM clause
    from_clause = "read_csv_auto(?, header=True)"
    
    # Build WHERE clause
    where_parts = []
    for filter_item in plan.get("filters", []):
        col = filter_item.get("col")
        op = filter_item.get("op")
        val = filter_item.get("val")
        
        if col and op:
            # Handle different value types
            if isinstance(val, str):
                if op == "contains":
                    where_parts.append(f'"{col}" LIKE \'%{val}%\'')
                else:
                    where_parts.append(f'"{col}" {op} \'{val}\'')
            else:
                where_parts.append(f'"{col}" {op} {val}')
    
    where_clause = ""
    if where_parts:
        where_clause = "WHERE " + " AND ".join(where_parts)
    
    # Build GROUP BY clause
    groupby_clause = ""
    groupby_cols = plan.get("groupby", [])
    if groupby_cols:
        groupby_clause = "GROUP BY " + ", ".join([f'"{col}"' for col in groupby_cols])
    
    # Build ORDER BY clause
    orderby_clause = ""
    sort_items = plan.get("sort", [])
    if sort_items:
        sort_parts = []
        for sort_item in sort_items:
            col = sort_item.get("col")
            dir = sort_item.get("dir", "asc").upper()
            sort_parts.append(f'"{col}" {dir}')
        orderby_clause = "ORDER BY " + ", ".join(sort_parts)
    
    # Build LIMIT clause
    limit_clause = ""
    limit_val = plan.get("limit")
    if limit_val:
        limit_clause = f"LIMIT {limit_val}"
    
    # Combine all clauses
    sql_parts = [
        f"SELECT {select_clause}",
        f"FROM {from_clause}",
        where_clause,
        groupby_clause,
        orderby_clause,
        limit_clause
    ]
    
    sql = " ".join([part for part in sql_parts if part])
    return sql

def run_plan_on_csv(local_csv_path: str, plan: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a plan against a CSV file using DuckDB.
    Returns SQL and table results.
    Reuses DuckDB connection pattern from run_fixed_duckdb_query.
    """
    sql = plan_to_sql(plan)
    
    try:
        con = duckdb.connect(database=":memory:")
        df = con.execute(sql, [local_csv_path]).df()
        con.close()
        
        return {
            "sql": sql,
            "table": {
                "columns": list(df.columns),
                "rows": df.to_dict(orient='records'),
                "rows_returned": len(df)
            }
        }
    except Exception as e:
        logger.error(f"Error executing plan SQL: {str(e)}")
        raise ValueError(f"Failed to execute plan: {str(e)}")

def llm_generate_plan(question: str, planner_schema: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate a plan from a question using LLM.
    Converts planner_schema to column_metadata format for query_to_plan.
    Reuses query_to_plan and planner_llm_call.
    """
    # Convert planner_schema back to column_metadata format
    column_metadata = {}
    for col_info in planner_schema:
        col_name = col_info.get("name")
        if col_name:
            column_metadata[col_name] = {
                "canon_type": col_info.get("canon_type", "text"),
                "examples": col_info.get("examples", [])
            }
    
    return query_to_plan(question, column_metadata, llm_call=planner_llm_call)

@app.route("/api/table/execute_plan", methods=["POST"])
@require_solari_key
def execute_plan_endpoint():
    t0 = time.time()
    body = request.get_json(force=True) or {}

    team_id = body.get("team_id") or body.get("teamId")
    user_id = body.get("user_id")
    source_id = body.get("source_id")   # Firestore doc id for the table
    agent_id = body.get("agent_id")
    raw_plan = body.get("plan")

    if not team_id or not user_id or not source_id or not agent_id or not isinstance(raw_plan, dict):
        return jsonify({
            "success": False,
            "error": "Required: team_id, user_id, source_id, agent_id, plan (object)"
        }), 400

    try:
        # 1) Read schema/meta from Firestore
        table_meta = get_table_meta(team_id, user_id, source_id, agent_id)
        planner_schema = build_planner_schema_from_table_meta(table_meta, max_examples=3)
        col_types = col_types_from_planner_schema(planner_schema)

        # 2) Normalize + validate (+ fuzzy fix once)
        filtered_plan = _filter_allowed_plan_keys(raw_plan)
        normalized_plan = normalize_plan_minimal(filtered_plan)

        vf = validate_then_fix_once_with_sanity(normalized_plan, col_types)
        if not vf["success"]:
            vf["ms"] = int((time.time() - t0) * 1000)
            vf["planner_schema"] = planner_schema
            vf["normalized_plan"] = normalized_plan
            return jsonify(vf), 200

        final_plan = vf["final_plan"]

        # 3) Download CSV to temp + run DuckDB
        local_csv_path = download_table_csv_to_temp(team_id, user_id, source_id, agent_id)
        out = run_plan_on_csv(local_csv_path, final_plan)

        return jsonify({
            "success": True,
            "ms": int((time.time() - t0) * 1000),
            "planner_schema": planner_schema,
            "normalized_plan": normalized_plan,
            "final_plan": final_plan,
            "fixes_applied": vf["fixes_applied"],
            "issues": [],
            "suggestions": vf["suggestions"],
            "sql": out["sql"],
            "table": out["table"],
        }), 200

    except Exception as e:
        logger.error(f"Error executing plan: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e),
            "ms": int((time.time() - t0) * 1000),
        }), 500

def query_table_endpoint_internal(
    team_id: str,
    user_id: str,
    source_id: str,
    agent_id: str,
    question: str,
    request_id: str | None = None,
    suggested_source: str | None = None,
):
    """
    Internal function to handle table queries (extracted from query_table_endpoint).
    This allows the table query logic to be called from other endpoints.
    
    Args:
        user_id: User identifier
        source_id: Source document ID
        agent_id: Agent identifier
        question: User's question
    
    Returns:
        Flask JSON response with table query results
    """
    t0 = time.time()
    
    try:
        # 1) Read schema/meta from Firestore
        table_meta = get_table_meta(team_id, user_id, source_id, agent_id)
        planner_schema = build_planner_schema_from_table_meta(table_meta, max_examples=3)
        col_types = col_types_from_planner_schema(planner_schema)

        # 2) LLM plan (raw)
        raw_plan = llm_generate_plan(question=question, planner_schema=planner_schema)

        # 3) Normalize + validate (+ fuzzy fix once)
        filtered_plan = _filter_allowed_plan_keys(raw_plan)
        normalized_plan = normalize_plan_minimal(filtered_plan)

        vf = validate_then_fix_once_with_sanity(normalized_plan, col_types)
        if not vf["success"]:
            return jsonify({
                "success": False,
                "ms": int((time.time() - t0) * 1000),
                "planner_schema": planner_schema,
                "raw_plan": raw_plan,
                "normalized_plan": normalized_plan,
                "final_plan": vf["final_plan"],
                "fixes_applied": vf["fixes_applied"],
                "issues": vf["issues"],
                "suggestions": vf["suggestions"],
                "requestId": request_id,
                "suggestedSource": suggested_source,
            }), 200

        final_plan = vf["final_plan"]

        # 4) Download CSV to temp + run DuckDB
        local_csv_path = download_table_csv_to_temp(team_id, user_id, source_id, agent_id)
        out = run_plan_on_csv(local_csv_path, final_plan)

        # 5) Generate summarized response using OpenAI
        response_summarized = None
        try:
            # Format table data for the LLM
            table_data = out["table"]
            columns = table_data.get("columns", [])
            rows = table_data.get("rows", [])
            
            # Create a readable table representation
            table_text = f"Columns: {', '.join(columns)}\n\n"
            table_text += "Data:\n"
            for i, row in enumerate(rows[:20], 1):  # Limit to first 20 rows for context
                row_str = ", ".join([f"{col}: {row.get(col, 'N/A')}" for col in columns])
                table_text += f"{i}. {row_str}\n"
            
            if len(rows) > 20:
                table_text += f"\n... and {len(rows) - 20} more rows\n"
            
            # Create prompt for summarization
            summary_prompt = f"""Based on the following table data, provide a concise answer (maximum 3 sentences) to the user's question.

User's Question: {question}

Table Data:
{table_text}

Provide a clear, concise answer to the question based on the table data above."""
            
            logger.info("Generating summarized response using OpenAI...")
            openai_client = get_openai_client()
            summary_response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": summary_prompt}],
                temperature=0.3,
                max_tokens=150  # Limit tokens to keep it concise
            )
            response_summarized = summary_response.choices[0].message.content.strip()
            logger.info(f"Generated summary: {response_summarized[:100]}...")
        except Exception as e:
            logger.error(f"Error generating summarized response: {str(e)}", exc_info=True)
            # Don't fail the whole request if summarization fails
            response_summarized = None

        return jsonify({
            "success": True,
            "ms": int((time.time() - t0) * 1000),
            "planner_schema": planner_schema,
            "raw_plan": raw_plan,
            "normalized_plan": normalized_plan,
            "final_plan": final_plan,
            "fixes_applied": vf["fixes_applied"],
            "issues": [],
            "suggestions": vf["suggestions"],
            "sql": out["sql"],
            "table": out["table"],
            "response_summarized": response_summarized,
            "requestId": request_id,
            "suggestedSource": suggested_source,
        }), 200

    except Exception as e:
        logger.error(f"Error querying table: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e),
            "ms": int((time.time() - t0) * 1000),
            "requestId": request_id,
            "suggestedSource": suggested_source,
        }), 500

@app.route("/api/table/ask/ai_sql", methods=["POST"])
@require_solari_key
def query_table_endpoint():
    body = request.get_json(force=True) or {}

    team_id = body.get("team_id") or body.get("teamId")
    user_id = body.get("user_id")
    source_id = body.get("source_id")
    agent_id = body.get("agent_id")
    question = body.get("question")

    if not team_id or not user_id or not source_id or not agent_id or not question:
        return jsonify({
            "success": False,
            "error": "Required: team_id, user_id, source_id, agent_id, question"
        }), 400

    return query_table_endpoint_internal(team_id, user_id, source_id, agent_id, question, request_id=body.get("requestId"))

@app.route("/api/table/plan/check", methods=["POST"])
@require_solari_key
def check_table_plan():
    """
    Sanity check a table plan against Firestore metadata.
    
    Expected request body:
    {
      "team_id": "...",
      "user_id": "...",
      "document_id": "...",
      "agent_id": "...",
      "plan": {...}  # OR "llm_output": {"plan": {...}, ...}
    }
    
    Returns:
        JSON response with validation results, issues, suggestions, and normalized plan
    """
    t0 = time.time()
    body = request.get_json(force=True) or {}

    # Required identifiers
    team_id = body.get("team_id") or body.get("teamId")
    user_id = body.get("user_id")
    document_id = body.get("document_id")
    agent_id = body.get("agent_id")

    if not team_id or not user_id or not document_id or not agent_id:
        return jsonify({
            "success": False,
            "error": "Missing required fields: team_id, user_id, document_id, agent_id"
        }), 400

    # Accept either a raw plan or a planner-style llm_output object
    if "llm_output" in body and body["llm_output"]:
        llm_output = body["llm_output"]
    elif "plan" in body and body["plan"]:
        # Wrap as llm_output shape expected by your step8 checker
        raw_plan = body["plan"]
        llm_output = {"plan": raw_plan, "success": True}
    else:
        return jsonify({
            "success": False,
            "error": "Provide either 'plan' or 'llm_output' in the request body."
        }), 400

    try:
        # IMPORTANT: ensure only allowed keys are passed through to the checker
        raw_plan = llm_output.get("plan") or {}
        filtered_plan = {k: raw_plan[k] for k in raw_plan if k in ALLOWED_KEYS}
        llm_output = dict(llm_output)
        llm_output["plan"] = filtered_plan

        # Run your sanity check logic against Firestore metadata
        result = step8_sanity_check(
            team_id=team_id,
            user_id=user_id,
            document_id=document_id,
            agent_id=agent_id,
            llm_output=llm_output,
            get_table_meta_fn=get_table_meta,   # your function
        )

        result["ms"] = int((time.time() - t0) * 1000)
        return jsonify(result), 200

    except Exception as e:
        # If anything blows up, return the error + timing
        logger.error(f"Error checking table plan: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e),
            "ms": int((time.time() - t0) * 1000)
        }), 500


# rag related code

def perform_rag_query(
    userid: str,
    namespace: str,
    query: str,
    nickname: str = '',
    source_type: str = '',
    model: str = "gpt-4o-mini"
):
    """
    Perform RAG query against Pinecone and generate answer.
    
    Args:
        userid: User identifier
        namespace: Pinecone namespace
        query: User's query
        nickname: Optional nickname filter
        source_type: Optional source type filter
    
    Returns:
        dict with 'success', 'answer', 'metadata', and optionally 'error'
    """
    try:
        logger.info(f"Processing query for user: {userid}, namespace: {namespace}, query: {query[:100]}...")
        
        # Step 1: Generate embedding for the query
        logger.info("Generating embedding for query...")
        openai_client = get_openai_client()
        query_embedding = generate_embeddings([query], openai_client)[0]
        
        # Step 2: Query Pinecone using GRPC
        logger.info(f"Querying Pinecone namespace: {namespace}...")
        index = get_pinecone_grpc_index()
        
        # Prepare filter dictionary with proper $eq format
        filter_dict = {}
        if nickname:
            filter_dict['nickname'] = {"$eq": nickname}
        if source_type:
            filter_dict['source'] = {"$eq": source_type}
        
        # Execute query
        query_results = index.query(
            namespace=namespace,
            vector=query_embedding,
            top_k=10,
            filter=filter_dict if filter_dict else None,
            include_metadata=True,
            include_values=False
        )
        
        logger.info(f"Found {len(query_results.matches)} results")
        
        # Step 3: Extract text from matches for context
        context_chunks = []
        for match in query_results.matches:
            metadata = match.metadata or {}
            # Try to get full text, fallback to text_preview if available
            chunk_text = metadata.get('text') or metadata.get('text_preview') or ''
            if chunk_text:
                context_chunks.append(chunk_text)
        
        if not context_chunks:
            return {
                'success': False,
                'answer': None,
                'metadata': None,
                'error': 'No text content found in retrieved chunks'
            }
        
        # Combine context chunks
        context = "\n".join(context_chunks)
        logger.info(f"Extracted context with {len(context)} characters from {len(context_chunks)} chunks")
        
        # Step 4: Create augmented prompt
        prompt = f"""QUESTION:
{query}

CONTEXT:
{context}

Using the CONTEXT provided, answer the QUESTION. Keep your answer grounded in the facts of the CONTEXT. If the CONTEXT doesn't contain the answer to the QUESTION, say you don't know."""
        
        # Step 5: Send to LLM to generate answer
        logger.info("Generating answer using KeywordsAI...")
        response = keywordsai_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            user_id=userid
        )

        answer = response["choices"][0]["message"]["content"]
        logger.info(f"Generated answer: {answer[:100]}...")
        
        # Format results with metadata for reference
        results = []
        for match in query_results.matches:
            results.append({
                'id': match.id,
                'score': match.score,
                'metadata': match.metadata
            })
        
        return {
            'success': True,
            'answer': answer,
            'chosen_nickname': nickname if nickname else None,
            'metadata': {
                'userid': userid,
                'namespace': namespace,
                'query': query,
                'chunks_used': len(context_chunks),
                'context_length': len(context),
                'retrieved_chunks': results,
                'nickname_filter': nickname if nickname else None,
                'chosen_nickname': nickname if nickname else None,
                'source_type_filter': source_type if source_type else None
            }
        }
        
    except Exception as e:
        logger.error(f"Error in RAG query: {str(e)}", exc_info=True)
        return {
            'success': False,
            'answer': None,
            'metadata': None,
            'error': f'Failed to perform RAG query: {str(e)}'
        }

def build_source_selection_prompt(query: str, sources: list) -> dict:
    """
    Build the system and user prompts for source selection.
    
    Args:
        query: User's query
        sources: List of dicts with 'nickname' and 'description' keys
    
    Returns:
        dict with 'system_prompt' and 'user_prompt'
    """
    # Build the list of sources for the prompt
    sources_list = "\n".join([f"- {s['nickname']}: {s['description']}" for s in sources])
    
    system_prompt = """You are a source selection assistant. Your task is to analyze a user query and determine which source is most helpful based on the source descriptions provided.

You must respond with ONLY the nickname value of the most relevant source. Do not include any explanation, preamble, or additional text. Just the nickname.

If no source is clearly relevant, respond with the first source's nickname."""
    
    user_prompt = f"""User Query: {query}

Available Sources:
{sources_list}

Which source nickname is most helpful for answering this query? Respond with ONLY the nickname value."""
    
    return {
        'system_prompt': system_prompt,
        'user_prompt': user_prompt
    }

def select_source_with_openai(
    query: str,
    sources: list,
    user_id: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.3
) -> str:
    """
    Use OpenAI to select the most relevant source based on query and source descriptions.
    
    Args:
        query: User's query
        sources: List of dicts with 'nickname' and 'description' keys
        model: OpenAI model to use (default: "gpt-3.5-turbo")
        temperature: Temperature for OpenAI call (default: 0.3)
    
    Returns:
        Selected nickname string
    
    Raises:
        Exception: If OpenAI call fails
    """
    # Build prompts
    prompts = build_source_selection_prompt(query, sources)
    
    logger.info("Using KeywordsAI to determine best source...")
    response = keywordsai_chat_completion(
        messages=[
            {"role": "system", "content": prompts['system_prompt']},
            {"role": "user", "content": prompts['user_prompt']}
        ],
        model=model,
        user_id=user_id,
        temperature=temperature
    )
    
    selected_nickname = response["choices"][0]["message"]["content"].strip()
    logger.info(f"KeywordsAI selected nickname: {selected_nickname}")
    
    return selected_nickname

def validate_selected_nickname(selected_nickname: str, sources: list) -> str:
    """
    Validate that the selected nickname exists in the sources list.
    If invalid, return the first source's nickname as fallback.
    
    Args:
        selected_nickname: The nickname returned by OpenAI
        sources: List of dicts with 'nickname' and 'description' keys
    
    Returns:
        Validated nickname (either the selected one or fallback)
    """
    valid_nicknames = [s['nickname'] for s in sources]
    
    if selected_nickname not in valid_nicknames:
        logger.warning(f"Selected nickname '{selected_nickname}' not found in valid sources. Using first source as fallback.")
        return sources[0]['nickname']
    
    return selected_nickname

def decide_source(userid: str, agent_id: str, namespace: str, query: str, source_type: str = '', model: str = "gpt-4o-mini"):
    """
    Decide which source to use when no nickname is provided.
    
    Args:
        userid: User identifier
        agent_id: Agent identifier
        namespace: Pinecone namespace
        query: User's query
        source_type: Optional source type filter
    
    Returns:
        dict with 'success', 'nickname' (if successful), and optionally 'error'
    """
    try:
        db = firestore.client()
        team_id = get_team_id_for_uid(db, userid)
        logger.info(f"Deciding source for team: {team_id}, user: {userid}, agent: {agent_id}, namespace: {namespace}, query: {query[:100]}...")
        
        # Step 1: Get list of sources from Firestore
        sources_ref = (
            db.collection('teams').document(team_id)
              .collection('agents').document(agent_id)
              .collection('sources')
        )
        sources_docs = sources_ref.stream()
        
        sources = []
        for doc in sources_docs:
            doc_data = doc.to_dict()
            description = doc_data.get('description', '')
            nickname = doc_data.get('nickname', '')
            
            if nickname and description:  # Only include sources with both nickname and description
                sources.append({
                    'nickname': nickname,
                    'description': description
                })
        
        if not sources:
            return {
                'success': False,
                'nickname': None,
                'error': 'No sources found with both nickname and description'
            }
        
        logger.info(f"Found {len(sources)} sources with descriptions")
        
        # Step 2: Use OpenAI to select the best source
        selected_nickname = select_source_with_openai(
            query,
            sources,
            user_id=userid,
            model=model
        )
        
        # Step 3: Validate the selected nickname
        validated_nickname = validate_selected_nickname(selected_nickname, sources)
        
        logger.info(f"Final selected source nickname: {validated_nickname}")
        
        return {
            'success': True,
            'nickname': validated_nickname
        }
        
    except Exception as e:
        logger.error(f"Error deciding source: {str(e)}", exc_info=True)
        return {
            'success': False,
            'nickname': None,
            'error': f'Failed to decide source: {str(e)}'
        }

@app.route('/api/ask-pinecone', methods=['POST'])
@require_solari_key
def ask_pinecone():
    """
    Query Pinecone namespace with a user query.
    
    Expected request body:
    {
        "userid": "user123",
        "namespace": "your-namespace",
        "query": "your search query",
        "nickname": "optional-nickname",  # optional
        "source_type": "optional-source-type"  # optional
    }
    
    Process:
    1. Validate required parameters
    2. Generate embedding for the query using OpenAI
    3. Query Pinecone namespace
    4. Return results
    
    Returns:
        JSON response with query results
    """
    try:
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'answer': None,
                'metadata': None,
                'error': 'Request body is required'
            }), 400

        request_id = data.get('requestId')
        suggested_source = data.get('suggestedSource')
        if not request_id:
            return jsonify({
                'success': False,
                'answer': None,
                'metadata': None,
                'error': 'requestId parameter is required'
            }), 400
        
        # Validate required fields
        userid = data.get('userid')
        if not userid:
            return jsonify({
                'success': False,
                'answer': None,
                'metadata': None,
                'error': 'userid parameter is required',
                'requestId': request_id,
                'suggestedSource': suggested_source
            }), 400
        
        namespace = data.get('namespace')
        if not namespace:
            return jsonify({
                'success': False,
                'answer': None,
                'metadata': None,
                'error': 'namespace parameter is required'
            }), 400
        
        query = data.get('query')
        if not query:
            return jsonify({
                'success': False,
                'answer': None,
                'metadata': None,
                'error': 'query parameter is required',
                'requestId': request_id,
                'suggestedSource': suggested_source
            }), 400
        
        # Optional parameters
        nickname = data.get('nickname', '')
        source_type = data.get('source_type', '')
        model = "gpt-4o-mini"
        
        # Perform RAG query
        result = perform_rag_query(userid, namespace, query, nickname, source_type, model=model)
        
        if result['success']:
            return jsonify(result), 200
        else:
            status_code = 400 if 'error' in result else 500
            return jsonify(result), status_code
        
    except BadRequest as e:
        logger.error(f"JSON parsing error: {str(e)}")
        return jsonify({
            'success': False,
            'answer': None,
            'metadata': None,
            'error': f'Invalid JSON in request body: {str(e)}'
        }), 400
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return jsonify({
            'success': False,
            'answer': None,
            'metadata': None,
            'error': str(e)
        }), 400
    except Exception as e:
        logger.error(f"Error querying Pinecone: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'answer': None,
            'metadata': None,
            'error': f'Failed to query Pinecone: {str(e)}'
        }), 500

@app.route('/api/source-confirmed', methods=['POST'])
@require_solari_key
def source_confirmed():
    """
    Confirm source selection and perform RAG query or table query.
    
    Expected request body:
    {
        "requestId": "req-123",
        "suggestedSource": "optional-suggested-nickname",
        "userid": "user123",
        "agent_id": "agent123",  # required when nickname is provided
        "namespace": "your-namespace",  # optional, required only for RAG queries
        "query": "your search query",
        "nickname": "optional-nickname",  # optional
        "source_type": "optional-source-type"  # optional
    }
    
    Process:
    1. Validate required parameters
    2. If nickname provided, add query to source document's example_questions array
    3. Check source type - if table, route to table endpoint; otherwise RAG query
    4. Return results
    
    Returns:
        JSON response with RAG answer and metadata, or table query results
    """
    request_id = None
    suggested_source = None
    try:
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'answer': None,
                'metadata': None,
                'error': 'Request body is required',
                'requestId': request_id,
                'suggestedSource': suggested_source
            }), 400

        request_id = data.get('requestId')
        
        # Validate required fields
        userid = data.get('userid')
        if not userid:
            return jsonify({
                'success': False,
                'answer': None,
                'metadata': None,
                'error': 'userid parameter is required',
                'requestId': request_id
            }), 400
        
        query = data.get('query')
        if not query:
            return jsonify({
                'success': False,
                'answer': None,
                'metadata': None,
                'error': 'query parameter is required',
                'requestId': request_id
            }), 400
        
        # Optional parameters
        namespace = data.get('namespace', '')
        nickname = data.get('nickname', '')
        source_type = data.get('source_type', '')
        agent_id = data.get('agent_id')
        
        # agent_id is required when nickname is provided (needed to update source document)
        if nickname and not agent_id:
            return jsonify({
                'success': False,
                'answer': None,
                'metadata': None,
                'error': 'agent_id parameter is required when nickname is provided',
                'requestId': request_id,
                'suggestedSource': suggested_source
            }), 400
        
        try:
            db = firestore.client()
            team_id = get_team_id_for_uid(db, userid)
        except KeyError as e:
            return jsonify({
                'success': False,
                'answer': None,
                'metadata': None,
                'error': str(e),
                'requestId': request_id,
                'suggestedSource': suggested_source
            }), 404

        model = get_agent_model_provider(db, team_id, agent_id) if agent_id else "gpt-4o-mini"
        logger.info(f"Source confirmed for team: {team_id}, user: {userid}, agent: {agent_id}, namespace: {namespace}, nickname: {nickname}, query: {query[:100]}...")
        
        # If nickname is provided, check source type and route accordingly
        if nickname and agent_id:
            try:
                sources_ref = (
                    db.collection('teams').document(team_id)
                      .collection('agents').document(agent_id)
                      .collection('sources')
                )
                
                # Find the source document with matching nickname
                sources_docs = sources_ref.where('nickname', '==', nickname).stream()
                source_doc = None
                source_id = None
                for doc in sources_docs:
                    source_doc = doc
                    source_id = doc.id  # Document ID is the source_id
                    break
                
                if source_doc:
                    # Get current example_questions array or create empty one
                    doc_data = source_doc.to_dict()
                    example_questions = doc_data.get('example_questions', [])
                    
                    # Add query to array if not already present (avoid duplicates)
                    if query not in example_questions:
                        example_questions.append(query)
                        
                        # Update the document
                        source_doc.reference.update({
                            'example_questions': example_questions
                        })
                        logger.info(f"Added query to example_questions for source '{nickname}'. Total questions: {len(example_questions)}")
                    else:
                        logger.info(f"Query already exists in example_questions for source '{nickname}'")
                    
                    # Check if source is a table type
                    source_type_field = doc_data.get('type', '')
                    
                    if source_type_field == 'table':
                        logger.info(f"Source is a table, routing to /table/ask/ai_sql endpoint...")
                        # Route to table endpoint (namespace not needed for table queries)
                        return query_table_endpoint_internal(
                            team_id,
                            userid,
                            source_id,
                            agent_id,
                            query,
                            request_id=request_id,
                            suggested_source=suggested_source
                        )
                    else:
                        # Not a table, proceed with RAG query (namespace required)
                        if not namespace:
                            return jsonify({
                                'success': False,
                                'answer': None,
                                'metadata': None,
                                'error': 'namespace parameter is required for RAG queries',
                                'requestId': request_id,
                                'suggestedSource': suggested_source
                            }), 400
                        
                        logger.info(f"Source is not a table, running RAG query...")
                        result = perform_rag_query(userid, namespace, query, nickname, source_type, model=model)
                        
                        if result['success']:
                            result['requestId'] = request_id
                            result['suggestedSource'] = suggested_source
                            return jsonify(result), 200
                        else:
                            status_code = 400 if 'error' in result else 500
                            result['requestId'] = request_id
                            result['suggestedSource'] = suggested_source
                            return jsonify(result), status_code
                else:
                    logger.warning(f"Source document with nickname '{nickname}' not found for agent '{agent_id}'")
                    # Fall back to RAG query even if source not found (namespace required)
                    if not namespace:
                        return jsonify({
                            'success': False,
                            'answer': None,
                            'metadata': None,
                            'error': 'namespace parameter is required for RAG queries',
                            'requestId': request_id,
                            'suggestedSource': suggested_source
                        }), 400
                    
                    result = perform_rag_query(userid, namespace, query, nickname, source_type, model=model)
                    
                    if result['success']:
                        result['requestId'] = request_id
                        result['suggestedSource'] = suggested_source
                        return jsonify(result), 200
                    else:
                        status_code = 400 if 'error' in result else 500
                        result['requestId'] = request_id
                        result['suggestedSource'] = suggested_source
                        return jsonify(result), status_code
            except Exception as e:
                logger.error(f"Error updating source document or checking type: {str(e)}", exc_info=True)
                # Don't fail the request if updating example_questions fails, just log it and proceed with RAG
                if not namespace:
                    return jsonify({
                        'success': False,
                        'answer': None,
                        'metadata': None,
                        'error': 'namespace parameter is required for RAG queries',
                        'requestId': request_id,
                        'suggestedSource': suggested_source
                    }), 400
                
                result = perform_rag_query(userid, namespace, query, nickname, source_type, model=model)
                
                if result['success']:
                    result['requestId'] = request_id
                    result['suggestedSource'] = suggested_source
                    return jsonify(result), 200
                else:
                    status_code = 400 if 'error' in result else 500
                    result['requestId'] = request_id
                    result['suggestedSource'] = suggested_source
                    return jsonify(result), status_code
        else:
            # No nickname provided, just run RAG query without filtering (namespace required)
            if not namespace:
                return jsonify({
                    'success': False,
                    'answer': None,
                    'metadata': None,
                    'error': 'namespace parameter is required for RAG queries',
                    'requestId': request_id,
                    'suggestedSource': suggested_source
                }), 400
            
            result = perform_rag_query(userid, namespace, query, '', source_type, model=model)
            
            if result['success']:
                result['requestId'] = request_id
                result['suggestedSource'] = suggested_source
                return jsonify(result), 200
            else:
                status_code = 400 if 'error' in result else 500
                result['requestId'] = request_id
                result['suggestedSource'] = suggested_source
                return jsonify(result), status_code
        
    except BadRequest as e:
        logger.error(f"JSON parsing error: {str(e)}")
        return jsonify({
            'success': False,
            'answer': None,
            'metadata': None,
            'error': f'Invalid JSON in request body: {str(e)}',
            'requestId': request_id,
            'suggestedSource': suggested_source
        }), 400
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return jsonify({
            'success': False,
            'answer': None,
            'metadata': None,
            'error': str(e),
            'requestId': request_id,
            'suggestedSource': suggested_source
        }), 400
    except Exception as e:
        logger.error(f"Error in source-confirmed: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'answer': None,
            'metadata': None,
            'error': f'Failed to process source-confirmed request: {str(e)}',
            'requestId': request_id,
            'suggestedSource': suggested_source
        }), 500

@app.route('/api/handle-rag-message', methods=['POST'])
@require_solari_key
def handle_rag_message():
    """
    Handle RAG message with optional nickname routing.
    
    Expected request body:
    {
        "requestId": "req-123",
        "userid": "user123",
        "agent_id": "agent123",
        "namespace": "your-namespace",
        "query": "your search query",
        "nickname": "optional-nickname",  # optional
        "source_type": "optional-source-type"  # optional
    }
    
    Process:
    - If nickname is present: check source type
      - If table: route to /table/ask/ai_sql endpoint
      - If not table: run RAG query with that nickname
    - If no nickname: run decide_source function first
    
    Returns:
        JSON response with RAG answer and metadata, or table query results
    """
    request_id = None
    try:
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'answer': None,
                'metadata': None,
                'error': 'Request body is required',
                'requestId': request_id
            }), 400

        request_id = data.get('requestId')
        if not request_id:
            return jsonify({
                'success': False,
                'answer': None,
                'metadata': None,
                'error': 'requestId parameter is required',
                'requestId': request_id
            }), 400
        
        # Validate required fields
        userid = data.get('userid')
        if not userid:
            return jsonify({
                'success': False,
                'answer': None,
                'metadata': None,
                'error': 'userid parameter is required',
                'requestId': request_id
            }), 400
        
        namespace = data.get('namespace')
        if not namespace:
            return jsonify({
                'success': False,
                'answer': None,
                'metadata': None,
                'error': 'namespace parameter is required',
                'requestId': request_id
            }), 400
        
        query = data.get('query')
        if not query:
            return jsonify({
                'success': False,
                'answer': None,
                'metadata': None,
                'error': 'query parameter is required',
                'requestId': request_id
            }), 400
        
        agent_id = data.get('agent_id')
        if not agent_id:
            return jsonify({
                'success': False,
                'answer': None,
                'metadata': None,
                'error': 'agent_id parameter is required',
                'requestId': request_id
            }), 400
        
        # Optional parameters
        nickname = data.get('nickname', '')
        source_type = data.get('source_type', '')
        try:
            db = firestore.client()
            team_id = get_team_id_for_uid(db, userid)
        except KeyError as e:
            return jsonify({
                'success': False,
                'answer': None,
                'metadata': None,
                'error': str(e),
                'requestId': request_id
            }), 404
        
        model = get_agent_model_provider(db, team_id, agent_id)

        # If nickname is present, check source type and route accordingly
        if nickname:
            logger.info(f"Nickname provided ({nickname}), checking source type...")
            
            # Look up source document to check type
            try:
                sources_ref = (
                    db.collection('teams').document(team_id)
                      .collection('agents').document(agent_id)
                      .collection('sources')
                )
                
                # Find the source document with matching nickname
                sources_docs = sources_ref.where('nickname', '==', nickname).stream()
                source_doc = None
                source_id = None
                for doc in sources_docs:
                    source_doc = doc
                    source_id = doc.id  # Document ID is the source_id
                    break
                
                if not source_doc:
                    return jsonify({
                        'success': False,
                        'answer': None,
                        'metadata': None,
                        'error': f'Source with nickname "{nickname}" not found',
                        'requestId': request_id
                    }), 400
                
                # Check if source is a table type
                doc_data = source_doc.to_dict()
                source_type_field = doc_data.get('type', '')
                
                if source_type_field == 'table':
                    logger.info(f"Source is a table, routing to /table/ask/ai_sql endpoint...")
                    # Route to table endpoint
                    return query_table_endpoint_internal(team_id, userid, source_id, agent_id, query, request_id=request_id)
                else:
                    # Not a table, proceed with RAG query
                    logger.info(f"Source is not a table, running RAG query...")
                    result = perform_rag_query(userid, namespace, query, nickname, source_type, model=model)
                    
                    if result['success']:
                        result['requestId'] = request_id
                        return jsonify(result), 200
                    else:
                        result['requestId'] = request_id
                        return jsonify(result), 400
                        
            except Exception as e:
                logger.error(f"Error looking up source: {str(e)}", exc_info=True)
                return jsonify({
                    'success': False,
                    'answer': None,
                    'metadata': None,
                    'error': f'Failed to look up source: {str(e)}',
                    'requestId': request_id
                }), 500
        
        # If no nickname, run decide_source function first
        else:
            logger.info("No nickname provided, running decide_source function...")
            source_decision = decide_source(userid, agent_id, namespace, query, source_type, model=model)
            
            if not source_decision['success']:
                return jsonify({
                    'success': False,
                    'answer': None,
                    'metadata': None,
                    'error': source_decision.get('error', 'Failed to decide source'),
                    'requestId': request_id
                }), 400
            
            # Extract the selected nickname
            selected_nickname = source_decision.get('nickname')
            if not selected_nickname:
                return jsonify({
                    'success': False,
                    'answer': None,
                    'metadata': None,
                    'error': 'No nickname returned from decide_source',
                    'requestId': request_id
                }), 400
            
            logger.info(f"Selected nickname from decide_source: {selected_nickname}")
            
            # Return only the chosen nickname (don't call ask-pinecone yet)
            return jsonify({
                'success': True,
                'chosen_nickname': selected_nickname,
                'requestId': request_id,
                'metadata': {
                    'userid': userid,
                    'agent_id': agent_id,
                    'namespace': namespace,
                    'query': query,
                    'source_selection': {
                        'method': 'decide_source',
                        'selected_nickname': selected_nickname
                    }
                }
            }), 200
        
    except BadRequest as e:
        logger.error(f"JSON parsing error: {str(e)}")
        return jsonify({
            'success': False,
            'answer': None,
            'metadata': None,
            'error': f'Invalid JSON in request body: {str(e)}',
            'requestId': request_id
        }), 400
    except Exception as e:
        logger.error(f"Error handling RAG message: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'answer': None,
            'metadata': None,
            'error': f'Failed to handle RAG message: {str(e)}',
            'requestId': request_id
        }), 500

@app.route("/api/confluence/spaces", methods=["GET"])
@require_solari_key
def confluence_get_spaces():
    """
    Get all Confluence spaces for the authenticated user.
    
    Query Parameters:
        user_id (required): Firebase user ID
    
    Usage:
        GET /api/confluence/spaces?user_id=test_user_123
    
    Response:
        {
            "status": "success",
            "spaces": [...],
            "space_count": int
        }
    """
    user_id = request.args.get("user_id")
    
    if not user_id:
        return jsonify({"status": "failure", "error": "user_id is required"}), 400
    
    try:
        # Get credentials from Firestore (same as Jira - uses Atlassian OAuth)
        creds = _get_jira_creds(user_id)
        access_token = creds["access_token"]
        cloud_id = creds["cloud_id"]
        
        # Build Confluence API v2 URL
        url = f"https://api.atlassian.com/ex/confluence/{cloud_id}/wiki/api/v2/spaces"
        
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json"
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            logger.error(f"Failed to fetch Confluence spaces: {response.text}")
            return jsonify({
                "status": "failure",
                "error": f"Failed to fetch Confluence spaces: {response.text}"
            }), 400
        
        data = response.json()
        
        # Return all spaces
        spaces = data.get("results", [])
        return jsonify({
            "status": "success",
            "spaces": spaces,
            "space_count": len(spaces)
        }), 200
    
    except Exception as e:
        logger.error(f"Confluence spaces error: {str(e)}")
        return jsonify({
            "status": "failure",
            "error": str(e)
        }), 500


@app.route("/api/confluence/pages", methods=["GET"])
@require_solari_key
def confluence_get_space_pages():
    """
    Get pages from a specific Confluence space.
    
    Query Parameters:
        user_id (required): Firebase user ID
        space_id (required): Confluence space ID
    
    Usage:
        GET /api/confluence/pages?user_id=test_user_123&space_id=123456
    
    Response:
        {
            "status": "success",
            "pages": [...],
            "page_count": int
        }
    """
    user_id = request.args.get("user_id")
    space_id = request.args.get("space_id")
    
    if not user_id:
        return jsonify({"status": "failure", "error": "user_id is required"}), 400
    
    if not space_id:
        return jsonify({"status": "failure", "error": "space_id is required"}), 400
    
    try:
        # Get credentials from Firestore (same as Jira - uses Atlassian OAuth)
        creds = _get_jira_creds(user_id)
        access_token = creds["access_token"]
        cloud_id = creds["cloud_id"]
        
        # Build Confluence API v2 URL for pages in a specific space
        url = f"https://api.atlassian.com/ex/confluence/{cloud_id}/wiki/api/v2/spaces/{space_id}/pages"
        
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json"
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            logger.error(f"Failed to fetch Confluence pages: {response.text}")
            return jsonify({
                "status": "failure",
                "error": f"Failed to fetch Confluence pages: {response.text}"
            }), 400
        
        data = response.json()
        
        # Return pages
        pages = data.get("results", [])
        return jsonify({
            "status": "success",
            "pages": pages,
            "page_count": len(pages)
        }), 200
    
    except Exception as e:
        logger.error(f"Confluence pages error: {str(e)}")
        return jsonify({
            "status": "failure",
            "error": str(e)
        }), 500

@app.route("/api/confluence/page", methods=["GET"])
@require_solari_key
def confluence_get_page():
    """
    Get a specific Confluence page by page ID.
    
    Query Parameters:
        user_id (required): Firebase user ID
        page_id (required): Confluence page ID
    
    Usage:
        GET /api/confluence/page?user_id=test_user_123&page_id=123456
    
    Response:
        {
            "status": "success",
            "page": {...}
        }
    """
    user_id = request.args.get("user_id")
    page_id = request.args.get("page_id")
    
    if not user_id:
        return jsonify({"status": "failure", "error": "user_id is required"}), 400
    
    if not page_id:
        return jsonify({"status": "failure", "error": "page_id is required"}), 400
    
    try:
        # Get credentials from Firestore (same as Jira - uses Atlassian OAuth)
        creds = _get_jira_creds(user_id)
        access_token = creds["access_token"]
        cloud_id = creds["cloud_id"]
        
        # Build Confluence API v2 URL for a specific page
        url = f"https://api.atlassian.com/ex/confluence/{cloud_id}/wiki/api/v2/pages/{page_id}"
        
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json"
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            logger.error(f"Failed to fetch Confluence page: {response.text}")
            return jsonify({
                "status": "failure",
                "error": f"Failed to fetch Confluence page: {response.text}"
            }), 400
        
        page_data = response.json()
        
        return jsonify({
            "status": "success",
            "page": page_data
        }), 200
    
    except Exception as e:
        logger.error(f"Confluence page error: {str(e)}")
        return jsonify({
            "status": "failure",
            "error": str(e)
        }), 500

@app.route("/api/confluence/search", methods=["GET"])
@require_solari_key
def confluence_search_pages():
    """
    Search Confluence pages using CQL (Confluence Query Language).
    
    Query Parameters:
        user_id (required): Firebase user ID
        query (required): CQL query string (e.g., "type=page AND title~\"meeting\"")
        limit (optional): Maximum number of results (default: 10)
    
    Usage:
        GET /api/confluence/search?user_id=test_user_123&query=type=page%20AND%20title~%22meeting%22
        GET /api/confluence/search?user_id=test_user_123&query=type=page%20AND%20title~%22meeting%22&limit=20
    
    Response:
        {
            "status": "success",
            "results": [...],
            "result_count": int
        }
    """
    user_id = request.args.get("user_id")
    query = request.args.get("query")
    limit = request.args.get("limit", "10")
    
    if not user_id:
        return jsonify({"status": "failure", "error": "user_id is required"}), 400
    
    if not query:
        return jsonify({"status": "failure", "error": "query is required"}), 400
    
    try:
        # Get credentials from Firestore (same as Jira - uses Atlassian OAuth)
        creds = _get_jira_creds(user_id)
        access_token = creds["access_token"]
        cloud_id = creds["cloud_id"]
        
        # Build Confluence REST API search URL
        url = f"https://api.atlassian.com/ex/confluence/{cloud_id}/wiki/rest/api/search"
        
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json"
        }
        
        params = {
            "cql": query,
            "limit": limit
        }
        
        response = requests.get(url, headers=headers, params=params, timeout=45)
        
        if response.status_code != 200:
            logger.error(f"Failed to search Confluence pages: {response.text}")
            return jsonify({
                "status": "failure",
                "error": f"Failed to search Confluence pages: {response.text}"
            }), 400
        
        data = response.json()
        
        # Extract results from the response
        results = data.get("results", [])
        return jsonify({
            "status": "success",
            "results": results,
            "result_count": len(results)
        }), 200
    
    except Exception as e:
        logger.error(f"Confluence search error: {str(e)}")
        return jsonify({
            "status": "failure",
            "error": str(e)
        }), 500

@app.route("/api/confluence/get-page", methods=["GET"])
@require_solari_key
def confluence_get_full_page():
    """
    Get a specific Confluence page by page ID.
    
    Query Parameters:
        user_id (required): Firebase user ID
        page_id (required): Confluence page ID
        body_format (optional): Format for page body (e.g., "storage", "atlas_doc_format", "wiki"). Default: None
    
    Usage:
        GET /api/confluence/page?user_id=test_user_123&page_id=123456
        GET /api/confluence/page?user_id=test_user_123&page_id=123456&body_format=storage
    
    Response:
        {
            "status": "success",
            "page": {...}
        }
    """
    user_id = request.args.get("user_id")
    page_id = request.args.get("page_id")
    body_format = request.args.get("body_format")
    
    if not user_id:
        return jsonify({"status": "failure", "error": "user_id is required"}), 400
    
    if not page_id:
        return jsonify({"status": "failure", "error": "page_id is required"}), 400
    
    try:
        # Get credentials from Firestore (same as Jira - uses Atlassian OAuth)
        creds = _get_jira_creds(user_id)
        access_token = creds["access_token"]
        cloud_id = creds["cloud_id"]
        
        # Build Confluence API v2 URL for a specific page
        url = f"https://api.atlassian.com/ex/confluence/{cloud_id}/wiki/api/v2/pages/{page_id}"
        
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json"
        }
        
        # Add body-format (Confluence v2 expects hyphenated param)
        params = {}
        if body_format:
            params["body-format"] = body_format
        
        response = requests.get(url, headers=headers, params=params, timeout=10)
        
        if response.status_code != 200:
            logger.error(f"Failed to fetch Confluence page: {response.text}")
            return jsonify({
                "status": "failure",
                "error": f"Failed to fetch Confluence page: {response.text}"
            }), 400
        
        page_data = response.json()
        
        return jsonify({
            "status": "success",
            "page": page_data
        }), 200
    
    except Exception as e:
        logger.error(f"Confluence page error: {str(e)}")
        return jsonify({
            "status": "failure",
            "error": str(e)
        }), 500

def confluence_storage_html_to_rag_text(storage_html: str) -> str:
    """
    Convert Confluence 'storage' HTML into readable, RAG-friendly plaintext.
    Preserves headings, lists, tables. Removes noisy attributes.
    """
    if not storage_html or not storage_html.strip():
        return ""

    storage_html = html.unescape(storage_html)  # decode &ndash; etc.
    soup = BeautifulSoup(storage_html, "html.parser")

    # Remove noisy attributes (local-id, data-*, ac:*)
    for tag in soup.find_all(True):
        for attr in list(tag.attrs.keys()):
            if attr in ("href", "src"):
                continue
            if attr == "local-id" or attr.endswith("local-id") or attr.startswith("data-") or ":" in attr:
                tag.attrs.pop(attr, None)

    # Replace <br> with newline
    for br in soup.find_all("br"):
        br.replace_with("\n")

    lines: List[str] = []

    def add_line(s: str):
        s = s.strip()
        if s:
            lines.append(s)

    def render_node(node):
        if isinstance(node, NavigableString):
            txt = str(node).strip()
            if txt:
                add_line(txt)
            return

        if not isinstance(node, Tag):
            return

        name = node.name.lower()

        if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            level = int(name[1])
            text = node.get_text(" ", strip=True)
            add_line("#" * level + " " + text)
            add_line("")
            return

        if name == "hr":
            add_line("---")
            add_line("")
            return

        if name == "p":
            text = node.get_text(" ", strip=True)
            add_line(text)
            add_line("")
            return

        if name == "ul":
            for li in node.find_all("li", recursive=False):
                text = li.get_text(" ", strip=True)
                if text:
                    add_line(f"- {text}")
            add_line("")
            return

        if name == "ol":
            idx = int(node.get("start") or 1)
            for li in node.find_all("li", recursive=False):
                text = li.get_text(" ", strip=True)
                if text:
                    add_line(f"{idx}. {text}")
                    idx += 1
            add_line("")
            return

        if name == "table":
            rows = []
            for tr in node.find_all("tr"):
                cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
                if any(cells):
                    rows.append(cells)

            if rows:
                max_cols = max(len(r) for r in rows)
                rows = [r + [""] * (max_cols - len(r)) for r in rows]

                header = rows[0]
                add_line("| " + " | ".join(header) + " |")
                add_line("| " + " | ".join(["---"] * max_cols) + " |")
                for r in rows[1:]:
                    add_line("| " + " | ".join(r) + " |")
                add_line("")
            return

        for child in node.children:
            render_node(child)

    render_node(soup)

    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def upload_string_to_firebase_storage(storage_path: str, content: str, content_type: str = "text/plain") -> str:
    """
    Uploads a string to Firebase Storage at `storage_path`.
    Returns a gs:// URL (stable, good for storage).
    """
    bucket = storage.bucket()
    blob = bucket.blob(storage_path)
    blob.upload_from_string(content, content_type=content_type)
    return f"gs://{bucket.name}/{storage_path}"


# -------------------------
# Updated endpoint
# -------------------------

# @app.route("/api/confluence/add_pages", methods=["POST"])
# @require_solari_key
@app.route("/api/confluence/add_pages", methods=["POST"])
@require_solari_key
def confluence_add_pages():
    """
    Enqueue Confluence ingestion job (v1.5).
    """
    try:
        body = request.get_json(force=True) or {}

        user_id = body.get("user_id")
        agent_id = body.get("agent_id")
        pages = body.get("pages")

        logger.info(
            "Confluence add_pages request received",
            extra={"user_id": user_id, "agent_id": agent_id, "pages_count": len(pages or [])},
        )

        chunk_size = DEFAULT_CHUNK_SIZE
        chunk_overlap = DEFAULT_CHUNK_OVERLAP

        if not user_id:
            return jsonify({"status": "failure", "error": "user_id is required"}), 400
        if not agent_id:
            return jsonify({"status": "failure", "error": "agent_id is required"}), 400
        if not isinstance(pages, list) or not pages:
            return jsonify({"status": "failure", "error": "pages must be a non-empty list"}), 400

        db = firestore.client()
        team_id = get_team_id_for_uid(db, user_id)
        logger.info("Resolved team_id for user", extra={"user_id": user_id, "team_id": team_id})

        now = utcnow()
        # Build sources array
        sources = []
        for page in pages:
            if not isinstance(page, dict):
                return jsonify({"status": "failure", "error": "each page must be an object"}), 400

            page_id = page.get("id")
            title = page.get("title")
            tinyui_path = page.get("tinyui_path")
            url = page.get("url")
            excerpt = page.get("excerpt")

            if not page_id or not title:
                return jsonify({"status": "failure", "error": "each page must include id and title"}), 400

        sources.append({
                "source_key": f"confluence:{page_id}",
                "type": "confluence",
                "id": page_id,
                "title": title,
                "tinyui_path": tinyui_path,
                "url": url,
                "excerpt": excerpt,
                "status": "queued",
                "stage": "queued",
                "checkpoint": {
                    "chunk_index": 0,
                    "total_chunks": None,
                    "page_version": None,
                },
                "error": None,
            "updated_at": now,
            })

        expires_at = now + timedelta(days=30)

        job_id = uuid.uuid4().hex
        job_ref = (
            db.collection("teams").document(team_id)
              .collection("upload_jobs").document(job_id)
        )

        job_ref.set({
            "job_type": "ingest_sources",
            "connector": "confluence",
            "status": "queued",
            "created_at": firestore.SERVER_TIMESTAMP,
            "updated_at": firestore.SERVER_TIMESTAMP,
            "expires_at": expires_at,  # Firestore TTL field
            "locked_by": None,
            "locked_until": None,
            "progress": 0,
            "message": "Queued",
            "created_by_user_id": user_id,

            # context
            "team_id": team_id,
            "agent_id": agent_id,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,

            "sources": sources,
        })

        logger.info(
            "Confluence add_pages job queued",
            extra={"job_id": job_id, "team_id": team_id, "queued_sources": len(sources)},
        )

        return jsonify({
            "status": "success",
            "job_id": job_id,
            "team_id": team_id,
            "queued_sources": len(sources),
        }), 200
    except Exception as e:
        logger.exception("Confluence add_pages failed")
        return jsonify({"status": "failure", "error": str(e)}), 500
# def confluence_add_pages():
#     """
#     Add Confluence pages as sources for agent + team scopes, save raw+processed to Storage,
#     and upload processed text to Pinecone (chunked/embedded).

#     Request Body:
#     {
#       "user_id": "firebase_uid",
#       "team_id": "team_id",              # optional (derived if missing)
#       "agent_id": "agent_id",
#       "nickname": "confluence",          # optional, default "confluence" for now
#       "pages": [
#         {
#           "id": "327682",
#           "title": "2025-10-07 Meeting notes",
#           "tinyui_path": "/x/AgAF",
#           "excerpt": "optional preview text",
#           "url": "https://.../wiki/x/AgAF"
#         }
#       ],
#       "chunk_size": 1000,                # optional
#       "chunk_overlap": 200               # optional
#     }
#     """
#     try:
#         body = request.get_json(force=True) or {}

#         user_id = body.get("user_id")
#         team_id = body.get("team_id") or body.get("teamId")
#         agent_id = body.get("agent_id")
#         pages = body.get("pages")

#         chunk_size = int(body.get("chunk_size", 1000))
#         chunk_overlap = int(body.get("chunk_overlap", 200))

#         if not user_id:
#             return jsonify({"status": "failure", "error": "user_id is required"}), 400
#         if not agent_id:
#             return jsonify({"status": "failure", "error": "agent_id is required"}), 400
#         if not isinstance(pages, list) or not pages:
#             return jsonify({"status": "failure", "error": "pages must be a non-empty list"}), 400

#         db = firestore.client()
#         if not team_id:
#             team_id = get_team_id_for_uid(db, user_id)
#         namespace = _get_team_pinecone_namespace(db, team_id)

#         agent_sources_ref = (
#             db.collection("teams").document(team_id)
#               .collection("agents").document(agent_id)
#               .collection("sources")
#         )
#         team_sources_ref = db.collection("teams").document(team_id).collection("sources")

#         openai_client = get_openai_client()

#         added = 0
#         pinecone_uploaded_total = 0

#         for page in pages:
#             if not isinstance(page, dict):
#                 return jsonify({"status": "failure", "error": "each page must be an object"}), 400

#             page_id = page.get("id")
#             title = page.get("title")
#             tinyui_path = page.get("tinyui_path")
#             url = page.get("url")
#             body_preview = page.get("excerpt")
#             nickname = title

#             if not page_id or not title:
#                 return jsonify({"status": "failure", "error": "each page must include id and title"}), 400

#             logger.info(f"Adding Confluence page source: id={page_id}, title={title}, url={url}")

#             # 1) Fetch Confluence storage HTML via internal endpoint
#             storage_html = ""
#             try:
#                 base_url = request.host_url.rstrip("/")
#                 solari_key = request.headers.get(SOLARI_KEY_HEADER)
#                 storage_html = _fetch_confluence_page_storage_body(user_id, page_id, base_url, solari_key)
#             except Exception as e:
#                 logger.warning(f"Failed to fetch Confluence page body for {page_id}: {str(e)}")
#                 storage_html = ""

#             # 2) Convert to RAG-friendly text
#             processed_text = confluence_storage_html_to_rag_text(storage_html) if storage_html else ""

#             # 3) Upsert team-level source doc (and get its Firestore doc id = source_id)
#             team_match = team_sources_ref.where("id", "==", page_id).limit(1).stream()
#             team_doc = next(team_match, None)

#             if team_doc:
#                 team_source_id = team_doc.id
#                 team_sources_ref.document(team_source_id).update({
#                     "title": title,
#                     "id": page_id,
#                     "tinyui_path": tinyui_path,
#                     "url": url,
#                     "type": "confluence",
#                     "bodyPreview": body_preview,
#                     "nickname": nickname,
#                     "updatedAt": firestore.SERVER_TIMESTAMP,
#                     "agents": firestore.ArrayUnion([agent_id]),
#                     # NOTE: we no longer store raw html directly in Firestore (too big)
#                 })
#             else:
#                 new_ref = team_sources_ref.document()  # allocate id so we can use it in storage paths
#                 team_source_id = new_ref.id
#                 new_ref.set({
#                     "title": title,
#                     "id": page_id,
#                     "tinyui_path": tinyui_path,
#                     "url": url,
#                     "type": "confluence",
#                     "bodyPreview": body_preview,
#                     "nickname": nickname,
#                     "createdAt": firestore.SERVER_TIMESTAMP,
#                     "updatedAt": firestore.SERVER_TIMESTAMP,
#                     "agents": [agent_id],
#                 })

#             # 4) Upsert agent-level source doc (still keyed by Confluence page id)
#             agent_match = agent_sources_ref.where("id", "==", page_id).limit(1).stream()
#             agent_doc = next(agent_match, None)

#             agent_payload = {
#                 "title": title,
#                 "id": page_id,
#                 "tinyui_path": tinyui_path,
#                 "url": url,
#                 "type": "confluence_page",
#                 "bodyPreview": body_preview,
#                 "nickname": nickname,
#                 "updatedAt": firestore.SERVER_TIMESTAMP,
#             }

#             if agent_doc:
#                 agent_sources_ref.document(agent_doc.id).update(agent_payload)
#             else:
#                 agent_payload["createdAt"] = firestore.SERVER_TIMESTAMP
#                 agent_sources_ref.add(agent_payload)

#             # 5) Save raw + processed to Firebase Storage (under the TEAM source doc)
#             # Paths are deterministic and easy to find later.
#             raw_path = f"teams/{team_id}/sources/{team_source_id}/confluence/pages/{page_id}/raw_storage.html"
#             processed_path = f"teams/{team_id}/sources/{team_source_id}/confluence/pages/{page_id}/processed.txt"

#             raw_gs_url = ""
#             processed_gs_url = ""

#             if storage_html:
#                 raw_gs_url = upload_string_to_firebase_storage(raw_path, storage_html, content_type="text/html")

#             if processed_text:
#                 processed_gs_url = upload_string_to_firebase_storage(processed_path, processed_text, content_type="text/plain")

#             # Save pointers back to Firestore team source doc
#             team_sources_ref.document(team_source_id).update({
#                 "title": title,
#                 "id": page_id,
#                 "tinyui_path": tinyui_path,
#                 "url": url,
#                 "type": "confluence",
#                 "bodyPreview": body_preview,
#                 "nickname": nickname,
#                 "rawStoragePath": raw_path if storage_html else None,
#                 "rawStorageUrl": raw_gs_url if raw_gs_url else None,
#                 "processedTextPath": processed_path if processed_text else None,
#                 "processedTextUrl": processed_gs_url if processed_gs_url else None,
#                 "rawPath": raw_path if storage_html else None,
#                 "processedPath": processed_path if processed_text else None,
#                 "updatedAt": firestore.SERVER_TIMESTAMP,
#             })

#             # Save pointers to agent source doc as well
#             if agent_doc:
#                 agent_doc_id = agent_doc.id
#             else:
#                 # If agent doc was just created, it won't have id here; re-query
#                 agent_match_latest = agent_sources_ref.where("id", "==", page_id).limit(1).stream()
#                 agent_doc_latest = next(agent_match_latest, None)
#                 agent_doc_id = agent_doc_latest.id if agent_doc_latest else None

#             if agent_doc_id:
#                 agent_sources_ref.document(agent_doc_id).update({
#                     "title": title,
#                     "id": page_id,
#                     "tinyui_path": tinyui_path,
#                     "url": url,
#                     "type": "confluence_page",
#                     "bodyPreview": body_preview,
#                     "nickname": nickname,
#                     "rawStoragePath": raw_path if storage_html else None,
#                     "rawStorageUrl": raw_gs_url if raw_gs_url else None,
#                     "processedTextPath": processed_path if processed_text else None,
#                     "processedTextUrl": processed_gs_url if processed_gs_url else None,
#                     "rawPath": raw_path if storage_html else None,
#                     "processedPath": processed_path if processed_text else None,
#                     "updatedAt": firestore.SERVER_TIMESTAMP,
#                 })

#             # 6) Chunk/embed processed text and upload to Pinecone
#             if processed_text.strip():
#                 text_chunks = chunk_text(processed_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
#                 embeddings = generate_embeddings(text_chunks, openai_client)

#                 vectors = []
#                 for i, (chunk, emb) in enumerate(zip(text_chunks, embeddings)):
#                     vectors.append({
#                         "id": f"confluence_{team_source_id}_{page_id}_chunk_{i}",
#                         "values": emb,
#                         "metadata": {
#                             "source": "confluence",
#                             "nickname": nickname,
#                             "confluence_page_id": page_id,
#                             "source_id": team_source_id,  # Firestore doc id (team source doc)
#                             "text_preview": chunk[:500],
#                         }
#                     })

#                 uploaded = upload_vectors_to_pinecone(
#                     vectors=vectors,
#                     namespace=namespace,
#                     index_name="production",
#                     batch_size=100
#                 )
#                 pinecone_uploaded_total += uploaded
#             else:
#                 logger.info(f"No processed text for Confluence page {page_id}; skipping Pinecone upload.")

#             added += 1

#         return jsonify({
#             "status": "success",
#             "pages_added": added,
#             "pinecone_vectors_uploaded": pinecone_uploaded_total,
#             "namespace": namespace,
#             "nickname": nickname,
#         }), 200

#     except KeyError as e:
#         logger.error(f"Confluence add pages error: {str(e)}")
#         return jsonify({"status": "failure", "error": str(e)}), 400
#     except Exception as e:
#         logger.error(f"Confluence add pages error: {str(e)}", exc_info=True)
#         return jsonify({"status": "failure", "error": f"Internal server error: {str(e)}"}), 500

@app.route("/api/confluence/delete_page", methods=["POST"])
@require_solari_key
def confluence_delete_page():
    """
    Delete a Confluence page source from Firestore (agent + team) and Pinecone.
    
    Request Body:
    {
      "user_id": "firebase_uid",
      "agent_id": "agent_id",
      "page_id": "327682",
      "nickname": "2025-10-07 Meeting notes"  # optional, used for Pinecone filter
    }
    """
    try:
        body = request.get_json(force=True) or {}
        user_id = body.get("user_id")
        agent_id = body.get("agent_id")
        page_id = body.get("page_id") or body.get("id")
        nickname = body.get("nickname")
        
        if not user_id:
            return jsonify({"status": "failure", "error": "user_id is required"}), 400
        if not agent_id:
            return jsonify({"status": "failure", "error": "agent_id is required"}), 400
        if not page_id:
            return jsonify({"status": "failure", "error": "page_id is required"}), 400
        
        db = firestore.client()
        team_id = get_team_id_for_uid(db, user_id)
        namespace = _get_team_pinecone_namespace(db, team_id)
        
        removed_agent = False
        removed_team = False
        
        # --- Agent-level source delete ---
        agent_sources_ref = (
            db.collection("teams").document(team_id)
              .collection("agents").document(agent_id)
              .collection("sources")
        )
        agent_match = agent_sources_ref.where("id", "==", page_id).limit(1).stream()
        agent_doc = next(agent_match, None)
        if agent_doc:
            agent_sources_ref.document(agent_doc.id).delete()
            removed_agent = True
        
        # --- Team-level source delete ---
        team_sources_ref = db.collection("teams").document(team_id).collection("sources")
        team_match = team_sources_ref.where("id", "==", page_id).limit(1).stream()
        team_doc = next(team_match, None)
        if team_doc:
            team_sources_ref.document(team_doc.id).delete()
            removed_team = True
        
        # --- Pinecone delete ---
        filter_dict = {"confluence_page_id": {"$eq": page_id}}
        if nickname:
            filter_dict["nickname"] = {"$eq": nickname}
        
        pc = get_pinecone_client()
        index = pc.Index("production")
        index.delete(namespace=namespace, filter=filter_dict)
        
        return jsonify({
            "status": "success",
            "removed_agent": removed_agent,
            "removed_team": removed_team,
            "pinecone_filter": filter_dict,
        }), 200
    
    except KeyError as e:
        logger.error(f"Confluence delete page error: {str(e)}")
        return jsonify({"status": "failure", "error": str(e)}), 400
    except Exception as e:
        logger.error(f"Confluence delete page error: {str(e)}", exc_info=True)
        return jsonify({"status": "failure", "error": f"Internal server error: {str(e)}"}), 500

# @app.route("/api/confluence/add_pages", methods=["POST"])
# @require_solari_key
# def confluence_add_pages():
#     """
#     Add Confluence pages as sources for both agent and team scopes.
    
#     Request Body:
#     {
#       "user_id": "firebase_uid",
#       "team_id": "team_id",  # optional (derived from user if missing)
#       "agent_id": "agent_id",
#       "pages": [
#         {
#           "id": "163945",
#           "title": "Overview",
#           "tinyui_path": "/x/aYAC",
#           "excerpt": "optional preview text",
#           "url": "https://your-site.atlassian.net/wiki/x/aYAC"
#         }
#       ]
#     }
#     """
#     try:
#         body = request.get_json(force=True) or {}
#         user_id = body.get("user_id")
#         team_id = body.get("team_id") or body.get("teamId")
#         agent_id = body.get("agent_id")
#         pages = body.get("pages")
        
#         if not user_id:
#             return jsonify({"status": "failure", "error": "user_id is required"}), 400
#         if not agent_id:
#             return jsonify({"status": "failure", "error": "agent_id is required"}), 400
#         if not isinstance(pages, list) or not pages:
#             return jsonify({"status": "failure", "error": "pages must be a non-empty list"}), 400
        
#         db = firestore.client()
#         if not team_id:
#             team_id = get_team_id_for_uid(db, user_id)
        
#         agent_sources_ref = (
#             db.collection("teams").document(team_id)
#               .collection("agents").document(agent_id)
#               .collection("sources")
#         )
#         team_sources_ref = db.collection("teams").document(team_id).collection("sources")
        
#         added = 0
#         for page in pages:
#             if not isinstance(page, dict):
#                 return jsonify({"status": "failure", "error": "each page must be an object"}), 400
            
#             page_id = page.get("id")
#             title = page.get("title")
#             tinyui_path = page.get("tinyui_path")
#             url = page.get("url")
#             body_preview = page.get("excerpt")
            
#             if not page_id or not title:
#                 return jsonify({"status": "failure", "error": "each page must include id and title"}), 400

#             logger.info(f"Adding Confluence page source: id={page_id}, title={title}, url={url}")

#             body_storage_value = ""
#             try:
#                 body_storage_value = _fetch_confluence_page_storage_body(user_id, page_id)
#             except Exception as e:
#                 logger.warning(f"Failed to fetch Confluence page body for {page_id}: {str(e)}")
            
#             agent_payload = {
#                 "title": title,
#                 "id": page_id,
#                 "tinyui_path": tinyui_path,
#                 "url": url,
#                 "type": "confluence_page",
#                 "bodyPreview": body_preview,
#                 "bodyStorage": body_storage_value,
#                 "updatedAt": firestore.SERVER_TIMESTAMP,
#             }
#             team_payload = {
#                 "title": title,
#                 "id": page_id,
#                 "tinyui_path": tinyui_path,
#                 "url": url,
#                 "type": "confluence",
#                 "bodyPreview": body_preview,
#                 "bodyStorage": body_storage_value,
#                 "updatedAt": firestore.SERVER_TIMESTAMP,
#             }

#             # Agent-level source doc (upsert by page id)
#             agent_match = agent_sources_ref.where("id", "==", page_id).limit(1).stream()
#             agent_doc = next(agent_match, None)
#             if agent_doc:
#                 agent_sources_ref.document(agent_doc.id).update(agent_payload)
#             else:
#                 agent_payload["createdAt"] = firestore.SERVER_TIMESTAMP
#                 agent_sources_ref.add(agent_payload)
            
#             # Team-level source doc (upsert by page id)
#             team_match = team_sources_ref.where("id", "==", page_id).limit(1).stream()
#             team_doc = next(team_match, None)
#             if team_doc:
#                 team_sources_ref.document(team_doc.id).update({
#                     **team_payload,
#                     "agents": firestore.ArrayUnion([agent_id]),
#                 })
#             else:
#                 team_payload["agents"] = [agent_id]
#                 team_payload["createdAt"] = firestore.SERVER_TIMESTAMP
#                 team_sources_ref.add(team_payload)
            
#             added += 1
        
#         return jsonify({"status": "success", "pages_added": added}), 200
    
#     except KeyError as e:
#         logger.error(f"Confluence add pages error: {str(e)}")
#         return jsonify({"status": "failure", "error": str(e)}), 400
#     except Exception as e:
#         logger.error(f"Confluence add pages error: {str(e)}", exc_info=True)
#         return jsonify({"status": "failure", "error": f"Internal server error: {str(e)}"}), 500


# SLACK INTEGRATION WORK 

SLACK_CLIENT_ID = os.environ["SLACK_CLIENT_ID"]
SLACK_CLIENT_SECRET = os.environ["SLACK_CLIENT_SECRET"]
SLACK_REDIRECT_URI = os.environ["SLACK_REDIRECT_URI"]

# Your scopes (comma-separated string)
SLACK_SCOPES = ",".join([
    "app_mentions:read",
    "assistant:write",
    "channels:history",
    "channels:join",
    "channels:read",
    "commands",
    "groups:history",
    "groups:read", 
    "users:read"
])

def get_team_id_for_uid(db, uid: str) -> str:
    user_snap = db.collection("users").document(uid).get()
    if not user_snap.exists:
        raise KeyError("user_not_found")
    team_id = (user_snap.to_dict() or {}).get("teamId")
    if not team_id:
        raise KeyError("team_id_not_found")
    return str(team_id)

def get_team_user_ref(db, uid: str):
    team_id = get_team_id_for_uid(db, uid)
    team_user_ref = db.collection("teams").document(team_id).collection("users").document(uid)
    if not team_user_ref.get().exists:
        raise KeyError("user_not_found")
    return team_user_ref, team_id

def get_slack_installations_ref(db, uid: str):
    team_user_ref, team_id = get_team_user_ref(db, uid)
    return team_user_ref.collection("slack_installations"), team_id

def build_slack_authorize_url(state: str) -> str:
    return (
        "https://slack.com/oauth/v2/authorize"
        f"?client_id={SLACK_CLIENT_ID}"
        f"&scope={SLACK_SCOPES}"
        f"&redirect_uri={SLACK_REDIRECT_URI}"
        f"&state={state}"
    )
@require_solari_key
@app.post("/slack/start_auth")
def slack_start_auth():
    db = firestore.client()
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"ok": False, "error": "Request body is required"}), 400
        
        uid = data.get('userid')
        if not uid:
            return jsonify({"ok": False, "error": "userid parameter is required"}), 400
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 401

    try:
        _, team_id = get_team_user_ref(db, uid)
    except KeyError as e:
        return jsonify({"ok": False, "error": str(e)}), 404

    state = secrets.token_urlsafe(24)
    created_at = int(time.time())

    db.collection("oauth_states").document(state).set({
        "uid": uid,
        "team_id": team_id,
        "created_at": created_at,
        "provider": "slack",
    })

    authorize_url = build_slack_authorize_url(state)

    return jsonify({
        "ok": True,
        "authorize_url": authorize_url,
        "state": state,  # keep for debugging; you can remove later
    })

@app.get("/slack/oauth/callback")
def slack_oauth_callback():
    db = firestore.client()
    state = request.args.get("state", "")
    code = request.args.get("code", "")
    error = request.args.get("error")

    if error:
        return f"Slack OAuth error: {error}", 400

    if not state or not code:
        return "Missing state or code", 400

    # 1. Validate state
    state_ref = db.collection("oauth_states").document(state)
    snap = state_ref.get()
    if not snap.exists:
        return "Invalid state", 400

    data = snap.to_dict() or {}
    uid = data.get("uid")
    created_at = int(data.get("created_at", 0))

    # Expire after 10 minutes
    if int(time.time()) - created_at > 600:
        state_ref.delete()
        return "State expired. Please retry.", 400

    # One-time use
    state_ref.delete()

    # 2. Exchange code -> token
    token_resp = requests.post(
        "https://slack.com/api/oauth.v2.access",
        data={
            "code": code,
            "redirect_uri": SLACK_REDIRECT_URI,
        },
        auth=(SLACK_CLIENT_ID, SLACK_CLIENT_SECRET),
        timeout=30,
    ).json()

    if not token_resp.get("ok"):
        print("Slack token exchange failed:", token_resp)
        return "Slack token exchange failed. Check logs.", 400

    # 3. Store installation
    bot_token = token_resp["access_token"]
    team = token_resp.get("team") or {}
    team_id = team.get("id")

    if not team_id:
        logger.error("Missing team.id in Slack token response")
        return jsonify({"ok": False, "error": "Invalid token response: missing team ID"}), 400

    # Convert to string (in case it's a number) - Slack IDs are usually strings anyway
    team_id_str = str(team_id)

    try:
        installs_ref, _ = get_slack_installations_ref(db, uid)
    except KeyError as e:
        return jsonify({"ok": False, "error": str(e)}), 404

    installs_ref.document(team_id_str).set({
        "slack_team": team,
        "slack_team_id": team_id_str,
        "slack_bot_token": bot_token,   # encrypt later
        "slack_scope": token_resp.get("scope"),
        "slack_installed_at": int(time.time()),
        "slack_provider": "slack",
    }, merge=True)

    # 4. Redirect user back to frontend settings page
    return redirect(
        "https://app.usesolari.ai/settings/slack_callback",
        code=302
    )

@app.get("/api/slack/auth_test")
def slack_auth_test_endpoint():
    """
    Call like:
      GET /slack/auth_test?uid=FIREBASE_UID
    Optional:
      GET /slack/auth_test?uid=FIREBASE_UID&team_id=T12345
    """

    uid = request.args.get("uid")
    team_id = request.args.get("team_id")  # optional
    db = firestore.client()
    if not uid:
        return jsonify({"ok": False, "error": "missing_uid"}), 400

    # 1) Load the Slack installation doc
    try:
        installs_ref, _ = get_slack_installations_ref(db, uid)
    except KeyError as e:
        return jsonify({"ok": False, "error": str(e)}), 404

    install_doc = None
    if team_id:
        snap = installs_ref.document(team_id).get()
        if not snap.exists:
            return jsonify({"ok": False, "error": "installation_not_found", "team_id": team_id}), 404
        install_doc = snap
    else:
        # If team_id not provided, grab the first installation doc
        snaps = list(installs_ref.limit(1).stream())
        if not snaps:
            return jsonify({"ok": False, "error": "no_slack_installation_found"}), 404
        install_doc = snaps[0]
        team_id = install_doc.id

    install_data = install_doc.to_dict() or {}
    bot_token = install_data.get("slack_bot_token")

    if not bot_token:
        return jsonify({"ok": False, "error": "bot_token_missing", "team_id": team_id}), 500

    # 2) Call Slack auth.test
    try:
        slack_resp = requests.post(
            "https://slack.com/api/auth.test",
            headers={"Authorization": f"Bearer {bot_token}"},
            timeout=30,
        ).json()
    except Exception as e:
        return jsonify({"ok": False, "error": "slack_request_failed", "details": str(e)}), 502

    # 3) Return Slack response + which installation we used
    return jsonify({
        "ok": True,
        "uid": uid,
        "team_id_used": team_id,
        "slack_auth_test": slack_resp,
    })

def slack_get(token: str, method: str, params: dict):
    r = requests.get(
        f"https://slack.com/api/{method}",
        headers={"Authorization": f"Bearer {token}"},
        params=params,
        timeout=30,
    )
    data = r.json()
    if not data.get("ok"):
        raise RuntimeError(data)
    return data

def get_bot_token(db, uid: str, team_id: str | None):
    installs_ref, _ = get_slack_installations_ref(db, uid)

    if team_id:
        snap = installs_ref.document(team_id).get()
        if not snap.exists:
            raise KeyError(f"installation_not_found:{team_id}")
        data = snap.to_dict() or {}
        return data.get("slack_bot_token"), team_id

    snaps = list(installs_ref.limit(1).stream())
    if not snaps:
        raise KeyError("no_slack_installation_found")
    data = snaps[0].to_dict() or {}
    return data.get("slack_bot_token"), snaps[0].id

@app.get("/api/slack/list_channels")
def slack_channels():
    """
    GET /api/slack/list_channels?uid=FIREBASE_UID
    Optional: &team_id=T123
    Returns public + private channels (as visible to the bot token).
    """
    uid = request.args.get("uid")
    team_id = request.args.get("team_id")

    db = firestore.client()

    if not uid:
        return jsonify({"ok": False, "error": "missing_uid"}), 400

    try:
        bot_token, team_id_used = get_bot_token(db, uid, team_id)
    except KeyError as e:
        return jsonify({"ok": False, "error": str(e)}), 404

    if not bot_token:
        return jsonify({"ok": False, "error": "bot_token_missing"}), 500

    types = "public_channel,private_channel"
    cursor = None
    channels = []

    try:
        while True:
            params = {"types": types, "limit": 200}
            if cursor:
                params["cursor"] = cursor

            data = slack_get(bot_token, "conversations.list", params)
            channels.extend(data.get("channels", []))

            cursor = (data.get("response_metadata") or {}).get("next_cursor") or ""
            if not cursor:
                break

    except RuntimeError as e:
        # e.args[0] is the Slack error dict we raised
        return jsonify({"ok": False, "error": "slack_api_error", "details": e.args[0]}), 502

    # Return a UI-friendly shape
    result = []
    for c in channels:
        result.append({
            "id": c["id"],
            "name": c.get("name"),
            "is_private": c.get("is_private"),
            "is_member": c.get("is_member"),  # whether the bot is in the channel
            "num_members": c.get("num_members"),
            "topic": (c.get("topic") or {}).get("value"),
        })

    return jsonify({
        "ok": True,
        "uid": uid,
        "team_id_used": team_id_used,
        "count": len(result),
        "channels": result,
    })


SLACK_IGNORED_SUBTYPES = {
    "channel_join",
    "channel_leave",
    "channel_topic",
    "channel_purpose",
    "channel_name",
    "channel_archive",
    "channel_unarchive",
    # "bot_message",  # uncomment if you want to hide bot messages too
}

def is_system_message(m: dict) -> bool:
    return (m.get("subtype") in SLACK_IGNORED_SUBTYPES)

def slack_ts_to_str(ts: str, tz_name: str = "America/New_York") -> str:
    # Slack ts: "seconds.microseconds" as string
    try:
        sec = float(ts)
        dt = datetime.fromtimestamp(sec, tz=ZoneInfo(tz_name))
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return ts

def slack_get(bot_token: str, method: str, params: dict) -> dict:
    resp = requests.get(
        f"https://slack.com/api/{method}",
        headers={"Authorization": f"Bearer {bot_token}"},
        params=params,
        timeout=30,
    ).json()
    if not resp.get("ok"):
        raise RuntimeError(resp)
    return resp

def slack_get_user_name(bot_token: str, user_id: str, cache: dict[str, str]) -> str:
    if not user_id:
        return "unknown"
    if user_id in cache:
        return cache[user_id]

    try:
        data = slack_get(bot_token, "users.info", {"user": user_id})
    except Exception:
        cache[user_id] = user_id
        return user_id

    user = data.get("user") or {}
    profile = user.get("profile") or {}
    display_name = (profile.get("display_name") or "").strip()
    real_name = (profile.get("real_name") or "").strip()
    name = (user.get("name") or "").strip()

    resolved = display_name or real_name or name or user_id
    cache[user_id] = resolved
    return resolved

def get_bot_token_for_uid(db, uid: str, team_id: str | None = None) -> tuple[str, str]:
    installs_ref, _ = get_slack_installations_ref(db, uid)

    if team_id:
        snap = installs_ref.document(team_id).get()
        if not snap.exists:
            raise KeyError(f"installation_not_found:{team_id}")
        data = snap.to_dict() or {}
        return data.get("slack_bot_token"), team_id

    snaps = list(installs_ref.limit(1).stream())
    if not snaps:
        raise KeyError("no_slack_installation_found")
    data = snaps[0].to_dict() or {}
    return data.get("slack_bot_token"), snaps[0].id

def slack_message_permalink(team_id: str, channel_id: str, message_ts: str) -> str:
    # Slack deep link that opens the channel/message in Slack
    return f"https://slack.com/app_redirect?team={team_id}&channel={channel_id}&message_ts={message_ts}"

def flatten_slack_transcript(transcript: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Flattens a Slack transcript object into a clean, chronological list of messages.

    Expected input shape:
    - Either the full wrapper containing `threads_json`
    - Or a raw `threads_json` object
    """
    threads_json = transcript.get("threads_json") if "threads_json" in transcript else transcript
    threads = threads_json.get("threads", []) or []

    messages: List[Dict[str, Any]] = []

    for thread in threads:
        thread_ts = thread.get("thread_ts")

        for msg in thread.get("messages", []) or []:
            text = (msg.get("text") or "").strip()
            ts = msg.get("ts")

            if not text or not ts:
                continue

            messages.append({
                "ts": ts,
                "thread_ts": thread_ts,
                "user_id": msg.get("user"),
                "user_name": msg.get("user_name") or msg.get("username"),
                "text": text,
                "permalink": msg.get("permalink"),
                "is_reply": bool(msg.get("is_reply", False)),
            })

    try:
        messages.sort(key=lambda m: float(m["ts"]))
    except Exception:
        pass

    return messages

def chunk_messages(
    messages: List[Dict[str, Any]],
    chunk_n: int = 20,
    overlap_n: int = 5
) -> List[List[Dict[str, Any]]]:
    """
    Chunk a list of messages into overlapping windows.
    """
    if chunk_n <= 0:
        raise ValueError("chunk_n must be > 0")
    if overlap_n >= chunk_n:
        raise ValueError("overlap_n must be < chunk_n")

    chunks: List[List[Dict[str, Any]]] = []
    i = 0

    while i < len(messages):
        chunk = messages[i:i + chunk_n]
        if not chunk:
            break

        chunks.append(chunk)
        i += chunk_n - overlap_n

    return chunks

def load_latest_slack_transcript_chunk(db, uid: str, agent_id: str, source_id: str) -> Dict[str, Any]:
    team_user_ref, team_id_solari = get_team_user_ref(db, uid)

    source_ref = (
        db.collection("teams").document(team_id_solari)
          .collection("agents").document(agent_id)
          .collection("sources").document(source_id)
    )
    if not source_ref.get().exists:
        raise KeyError("source_not_found")

    chunks_ref = source_ref.collection("transcript_chunks")

    def query_by_last_message_ts():
        q = chunks_ref.order_by("last_message_ts", direction=firestore.Query.DESCENDING)
        return list(q.limit(1).stream())

    def query_by_created_at_fallback():
        q = chunks_ref.order_by("created_at", direction=firestore.Query.DESCENDING)
        return list(q.limit(1).stream())

    try:
        chunk_snaps = query_by_last_message_ts()
    except Exception:
        chunk_snaps = query_by_created_at_fallback()

    if not chunk_snaps:
        raise KeyError("no_transcript_chunks_found")

    chunk = chunk_snaps[0].to_dict() or {}
    storage_path = chunk.get("storage_path_threads")
    if not storage_path:
        raise KeyError("missing_storage_path_threads")

    bucket = storage.bucket()
    blob = bucket.blob(storage_path)
    raw = blob.download_as_text(encoding="utf-8")
    threads_json = json.loads(raw)

    return {
        "team_id_solari": team_id_solari,
        "storage_path_threads": storage_path,
        "chunk": chunk,
        "threads_json": threads_json,
    }

def pinecone_slack_upload_internal(
    db,
    uid: str,
    agent_id: str,
    channel_id: str,
    channel_name: Optional[str],
    namespace: str,
    nickname: str = "",
    chunk_n: int = 20,
    overlap_n: int = 5,
    team_id: Optional[str] = None,
    source_id: Optional[str] = None,
) -> Dict[str, Any]:
    if not channel_id or not namespace:
        return {"ok": False, "error": "missing_channel_id_or_namespace"}

    if not channel_name:
        try:
            team_user_ref, team_id_solari = get_team_user_ref(db, uid)
            source_ref = (
                db.collection("teams").document(team_id_solari)
                  .collection("agents").document(agent_id)
                  .collection("sources").document(channel_id)
            )
            source_doc = source_ref.get()
            if source_doc.exists:
                channel_name = (source_doc.to_dict() or {}).get("name")
        except KeyError:
            channel_name = None

    if not channel_name:
        return {"ok": False, "error": "channel_name_required"}

    sync_result = _sync_channel_transcript_internal(
        uid=uid,
        agent_id=agent_id,
        channel_id=channel_id,
        channel_name=channel_name,
        team_id=team_id,
    )
    if not sync_result.get("ok"):
        return {"ok": False, "error": sync_result.get("error"), "details": sync_result}

    if nickname:
        try:
            team_user_ref, team_id_solari = get_team_user_ref(db, uid)
            source_ref = (
                db.collection("teams").document(team_id_solari)
                  .collection("agents").document(agent_id)
                  .collection("sources").document(channel_id)
            )
            source_ref.set({"nickname": nickname}, merge=True)
        except Exception as e:
            logger.warning(f"Failed to update slack source nickname: {str(e)}")

    transcript_payload = load_latest_slack_transcript_chunk(db, uid, agent_id, channel_id)
    threads_json = transcript_payload.get("threads_json") or {}
    chunk_meta = transcript_payload.get("chunk") or {}
    storage_path_threads = transcript_payload.get("storage_path_threads")

    channel_id = threads_json.get("channel_id") or channel_id
    channel_name = threads_json.get("channel_name") or channel_name or ""
    team_id_used = threads_json.get("team_id") or chunk_meta.get("team_id")
    sync_run_id = threads_json.get("sync_run_id") or chunk_meta.get("sync_run_id")

    messages = flatten_slack_transcript(threads_json)
    if not messages:
        return {"ok": False, "error": "no_messages_to_upload"}

    chunks = chunk_messages(messages, chunk_n=chunk_n, overlap_n=overlap_n)

    chunk_texts: List[str] = []
    chunk_metadatas: List[Dict[str, Any]] = []

    base_id_raw = f"slack_{team_id_used}_{channel_id}_{sync_run_id}"
    base_id = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in base_id_raw)
    resolved_source_id = source_id or channel_id

    for idx, chunk in enumerate(chunks):
        lines = [f"#channel: #{channel_name} ({channel_id})"]
        thread_ts_values = []
        ts_values = []

        for msg in chunk:
            ts = msg.get("ts")
            if ts:
                ts_values.append(float(ts))

            thread_ts = msg.get("thread_ts")
            if thread_ts:
                thread_ts_values.append(thread_ts)

            user_name = (msg.get("user_name") or msg.get("user_id") or "unknown").strip() or "unknown"
            text = msg.get("text") or ""
            prefix = "↳ " if msg.get("is_reply") else ""
            lines.append(f"{prefix}[{ts}] @{user_name}: {text}")

        min_ts = str(min(ts_values)) if ts_values else ""
        max_ts = str(max(ts_values)) if ts_values else ""
        thread_ts_list = sorted(set(thread_ts_values)) if thread_ts_values else []

        chunk_text = "\n".join(lines)
        chunk_texts.append(chunk_text)

        metadata = {
            "source": "slack",
            "uid": uid,
            "agent_id": agent_id,
            "source_id": resolved_source_id,
            "team_id": team_id_used,
            "channel_id": channel_id,
            "channel_name": channel_name,
            "sync_run_id": sync_run_id,
            "chunk_index": idx,
            "message_count": len(chunk),
            "thread_ts_list": thread_ts_list,
            "text_preview": chunk_text[:500],
            "storage_path_threads": storage_path_threads,
            "nickname": nickname,
        }
        if min_ts:
            metadata["min_ts"] = min_ts
        if max_ts:
            metadata["max_ts"] = max_ts

        chunk_metadatas.append(metadata)

    openai_client = get_openai_client()
    embeddings = generate_embeddings(chunk_texts, openai_client)

    vectors = []
    for i, (embedding, metadata) in enumerate(zip(embeddings, chunk_metadatas)):
        vectors.append({
            "id": f"{base_id}_chunk_{i}",
            "values": embedding,
            "metadata": metadata,
        })

    index_name = "production"
    total_uploaded = upload_vectors_to_pinecone(vectors, namespace, index_name=index_name, batch_size=100)

    return {
        "ok": True,
        "namespace": namespace,
        "index": index_name,
        "uid": uid,
        "agent_id": agent_id,
        "source_id": resolved_source_id,
        "channel_id": channel_id,
        "channel_name": channel_name,
        "sync_run_id": sync_run_id,
        "messages_processed": len(messages),
        "chunks_created": len(chunks),
        "vectors_uploaded": total_uploaded,
        "chunk_n": chunk_n,
        "overlap_n": overlap_n,
        "nickname": nickname,
    }

# --- endpoint ---

def _sync_channel_transcript_internal(uid: str, agent_id: str, channel_id: str, channel_name: str,
                                      team_id: Optional[str] = None, tz: str = "America/New_York",
                                      limit: int = 500):
    """
    Core sync logic for syncing a Slack channel transcript.
    Returns a dict with the result (not a Flask response).
    Catches all exceptions and returns error dicts.
    """
    try:
        if not uid or not agent_id or not channel_id or not channel_name:
            return {"ok": False, "error": "missing_uid_agent_id_channel_id_or_channel_name"}

        # 0) Load bot token
        try:
            bot_token, team_id_used = get_bot_token_for_uid(db, uid, team_id)
        except KeyError as e:
            return {"ok": False, "error": str(e)}
        if not bot_token:
            return {"ok": False, "error": "bot_token_missing"}

        try:
            _, team_id_solari = get_team_user_ref(db, uid)
        except KeyError as e:
            return {"ok": False, "error": str(e)}

        # 1) Source doc (Slack channel saved as a "source")
        source_ref = (
            db.collection("teams").document(team_id_solari)
              .collection("agents").document(agent_id)
              .collection("sources").document(channel_id)
        )

        source_snap = source_ref.get()
        existing = source_snap.to_dict() if source_snap.exists else {}

        last_message_ts = (existing.get("last_message_ts") or "").strip()
        thread_latest_reply_ts = existing.get("thread_latest_reply_ts") or {}

        # Oldest: prefer last_message_ts; fall back to last_synced_at (per your request)
        oldest = None
        if last_message_ts:
            oldest = last_message_ts
        else:
            last_synced_at = existing.get("last_synced_at")
            if last_synced_at:
                try:
                    oldest = str(last_synced_at.timestamp())
                except Exception:
                    oldest = None

        sync_run_id = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ") + "_" + uuid.uuid4().hex[:8]

        # 2) Fetch incremental history (paginate)
        cursor = None
        history_messages = []
        while True:
            params = {"channel": channel_id, "limit": min(200, limit - len(history_messages))}
            if oldest:
                params["oldest"] = oldest
                params["inclusive"] = True
            if cursor:
                params["cursor"] = cursor

            data = slack_get(bot_token, "conversations.history", params)
            history_messages.extend(data.get("messages", []) or [])

            if len(history_messages) >= limit:
                break

            cursor = (data.get("response_metadata") or {}).get("next_cursor") or ""
            if not cursor:
                break

        # 3) Clean history (system messages, empty text, dedupe by ts)
        cleaned = []
        seen_ts = set()
        for m in history_messages:
            ts_val = m.get("ts")
            if not ts_val or ts_val in seen_ts:
                continue
            seen_ts.add(ts_val)

            if is_system_message(m):
                continue

            text = (m.get("text") or "").strip()
            if not text:
                continue

            cleaned.append(m)

        cleaned.sort(key=lambda m: float(m.get("ts", "0")))

        # 3.5) Resolve Slack user IDs to display names (cached per sync run)
        user_name_cache: dict[str, str] = {}
        for m in cleaned:
            user_id = m.get("user") or ""
            if not user_id:
                continue
            m["user_name"] = slack_get_user_name(bot_token, user_id, user_name_cache)

        # 4) Identify threads to refresh (Level 2 via latest_reply vs map)
        threads_to_refresh = []
        for m in cleaned:
            ts_val = m.get("ts")
            thread_ts = m.get("thread_ts")
            reply_count = int(m.get("reply_count") or 0)
            latest_reply = (m.get("latest_reply") or "").strip()

            is_root = bool(thread_ts and ts_val and thread_ts == ts_val)
            if not (is_root and reply_count > 0 and latest_reply):
                continue

            prev_latest = (thread_latest_reply_ts.get(thread_ts) or "").strip()
            if (not prev_latest) or (float(latest_reply) > float(prev_latest)):
                threads_to_refresh.append(thread_ts)

        threads_to_refresh = sorted(set(threads_to_refresh), key=lambda x: float(x))

        # 5) Fetch replies for refreshed threads (attempt incremental via oldest)
        replies_by_thread_ts = {}
        for tts in threads_to_refresh:
            prev_latest = (thread_latest_reply_ts.get(tts) or "").strip()
            params = {"channel": channel_id, "ts": tts, "limit": 200}
            if prev_latest:
                params["oldest"] = prev_latest
                params["inclusive"] = True

            replies_by_thread_ts[tts] = slack_get(bot_token, "conversations.replies", params)

        # 6) Build transcript + threads.json object (for THIS sync run only)
        transcript_lines = [f"#channel: #{channel_name}  ({channel_id})", ""]
        threads_out = []  # list of {thread_ts, messages:[{ts,user,text,permalink,is_reply}]}

        max_ts_seen = float(last_message_ts) if last_message_ts else 0.0
        thread_latest_updates = {}

        for m in cleaned:
            ts_val = m.get("ts")
            if ts_val:
                max_ts_seen = max(max_ts_seen, float(ts_val))

            user = m.get("user") or "unknown"
            user_name = m.get("user_name") or user
            text = (m.get("text") or "").strip()

            thread_ts = m.get("thread_ts")
            reply_count = int(m.get("reply_count") or 0)
            latest_reply = (m.get("latest_reply") or "").strip()
            is_thread_root = bool(thread_ts and ts_val and thread_ts == ts_val and reply_count > 0)

            if is_thread_root:
                # root always included
                root_msg_obj = {
                    "ts": ts_val,
                    "user": user,
                    "user_name": user_name,
                    "text": text,
                    "is_reply": False,
                    "permalink": slack_message_permalink(team_id_used, channel_id, ts_val),
                }
                thread_msgs = [root_msg_obj]

                transcript_lines.append(f"[{slack_ts_to_str(ts_val, tz)}] @{user_name}: {text}")

                # track latest reply for metadata
                if latest_reply:
                    prev = (thread_latest_reply_ts.get(thread_ts) or "").strip()
                    if (not prev) or (float(latest_reply) > float(prev)):
                        thread_latest_updates[thread_ts] = latest_reply

                # add replies if fetched this run
                repl_payload = replies_by_thread_ts.get(thread_ts)
                if repl_payload and repl_payload.get("ok"):
                    repl_msgs = repl_payload.get("messages", []) or []
                    repl_msgs = [x for x in repl_msgs if not is_system_message(x) and (x.get("text") or "").strip()]
                    repl_msgs.sort(key=lambda x: float(x.get("ts", "0")))

                    for r in repl_msgs:
                        r_ts = r.get("ts")
                        if not r_ts:
                            continue
                        if r_ts == thread_ts:
                            continue  # root already included

                        max_ts_seen = max(max_ts_seen, float(r_ts))
                        r_user = r.get("user") or "unknown"
                        r_text = (r.get("text") or "").strip()
                        r_user_name = slack_get_user_name(bot_token, r_user, user_name_cache)

                        thread_msgs.append({
                            "ts": r_ts,
                            "user": r_user,
                            "user_name": r_user_name,
                            "text": r_text,
                            "is_reply": True,
                            "permalink": slack_message_permalink(team_id_used, channel_id, r_ts),
                        })

                        transcript_lines.append(
                            f"  ↳ [{slack_ts_to_str(r_ts, tz)}] @{r_user_name}: {r_text}"
                        )

                transcript_lines.append("")
                threads_out.append({"thread_ts": thread_ts, "messages": thread_msgs})

            else:
                # standalone message
                if thread_ts and ts_val and thread_ts != ts_val:
                    continue  # skip non-root reply entries if any appear

                transcript_lines.append(f"[{slack_ts_to_str(ts_val, tz)}] @{user_name}: {text}")
                transcript_lines.append("")

                threads_out.append({
                    "thread_ts": None,
                    "messages": [{
                        "ts": ts_val,
                        "user": user,
                        "user_name": user_name,
                        "text": text,
                        "is_reply": False,
                        "permalink": slack_message_permalink(team_id_used, channel_id, ts_val),
                    }]
                })

        transcript = "\n".join(transcript_lines).rstrip() + "\n"

        threads_json_obj = {
            "team_id": team_id_used,
            "channel_id": channel_id,
            "channel_name": channel_name,
            "sync_run_id": sync_run_id,
            "generated_at_unix": int(time.time()),
            "threads": threads_out,
        }

        # 7) Upload artifacts to Firebase Storage
        bucket = storage.bucket()  # default bucket from Firebase Admin init

        base_path = f"users/{uid}/agents/{agent_id}/slack/{team_id_used}/{channel_id}/sync_runs/{sync_run_id}"
        transcript_path = f"{base_path}/transcript.txt"
        threads_path = f"{base_path}/threads.json"

        bucket.blob(transcript_path).upload_from_string(transcript, content_type="text/plain; charset=utf-8")
        bucket.blob(threads_path).upload_from_string(
            json.dumps(threads_json_obj, ensure_ascii=False),
            content_type="application/json; charset=utf-8",
        )

        # 8) Write Firestore chunk index doc + update source metadata
        updated_thread_map = dict(thread_latest_reply_ts)
        updated_thread_map.update(thread_latest_updates)

        chunk_ref = source_ref.collection("transcript_chunks").document(sync_run_id)

        # Also create/update document in teams/{team_id}/sources with random ID
        user_source_id = secrets.token_urlsafe(16)  # Random document ID
        user_source_ref = db.collection("teams").document(team_id_solari).collection("sources").document(user_source_id)

        batch = db.batch()

        batch.set(
            source_ref,
            {
                "type": "slack_channel",
                "channel_id": channel_id,
                "name": channel_name,
                "nickname": channel_name,
                "team_id": team_id_used,

                "last_message_ts": str(max_ts_seen) if max_ts_seen else last_message_ts,
                "last_synced_at": firestore.SERVER_TIMESTAMP,
                "last_sync_run_id": sync_run_id,

                "thread_latest_reply_ts": updated_thread_map,
            },
            merge=True,
        )

        # Also update user-level source document (same structure + agent_id)
        batch.set(
            user_source_ref,
            {
                "type": "slack_channel",
                "channel_id": channel_id,
                "name": channel_name,
                "nickname": channel_name,
                "team_id": team_id_used,
                "agent_id": agent_id,  # Track which agent synced this

                "last_message_ts": str(max_ts_seen) if max_ts_seen else last_message_ts,
                "last_synced_at": firestore.SERVER_TIMESTAMP,
                "last_sync_run_id": sync_run_id,

                "thread_latest_reply_ts": updated_thread_map,
            },
            merge=True,
        )

        batch.set(
            chunk_ref,
            {
                "sync_run_id": sync_run_id,
                "created_at": firestore.SERVER_TIMESTAMP,
                "storage_path_transcript": transcript_path,
                "storage_path_threads": threads_path,
                "oldest_used": oldest,
                "last_message_ts": str(max_ts_seen) if max_ts_seen else last_message_ts,
                "message_count": len(cleaned),
                "thread_count": len(threads_to_refresh),
                "team_id": team_id_used,
                "channel_id": channel_id,
                "channel_name": channel_name,
            },
            merge=True,
        )

        batch.commit()

        # 9) Return small response (pointers + stats)
        return {
            "ok": True,
            "uid": uid,
            "agent_id": agent_id,
            "team_id_used": team_id_used,
            "channel_id": channel_id,
            "channel_name": channel_name,
            "sync_run_id": sync_run_id,
            "oldest_used": oldest,
            "message_count": len(cleaned),
            "threads_refreshed": len(threads_to_refresh),
            "storage": {
                "transcript_path": transcript_path,
                "threads_path": threads_path,
            },
        }
    except RuntimeError as e:
        # Slack API errors
        return {"ok": False, "error": "slack_api_error", "details": e.args[0]}
    except Exception as e:
        # All other exceptions
        logger.error(f"Error in _sync_channel_transcript_internal: {str(e)}", exc_info=True)
        return {"ok": False, "error": "sync_exception", "details": str(e)}

# endpoint + work to start a single channel sync/resync
@app.route("/slack/sync_channel_transcript",  methods=["POST"])
def slack_sync_channel_transcript():
    """
    GET /slack/sync_channel_transcript?uid=...&agent_id=...&channel_id=...&channel_name=...
    Optional: &team_id=...&tz=America/New_York&limit=500
    """
    uid = request.args.get("uid")
    agent_id = request.args.get("agent_id")
    channel_id = request.args.get("channel_id")
    channel_name = request.args.get("channel_name")
    team_id = request.args.get("team_id")
    tz = request.args.get("tz", "America/New_York")
    max_messages = int(request.args.get("limit", 500))

    result = _sync_channel_transcript_internal(
        uid=uid or "",
        agent_id=agent_id or "",
        channel_id=channel_id or "",
        channel_name=channel_name or "",
        team_id=team_id,
        tz=tz,
        limit=max_messages,
    )

    # Map error types to HTTP status codes
    if not result.get("ok"):
        error = result.get("error", "")
        if error == "missing_uid_agent_id_channel_id_or_channel_name":
            return jsonify(result), 400
        elif "KeyError" in str(error) or "not found" in str(error).lower():
            return jsonify(result), 404
        elif error == "slack_api_error":
            return jsonify(result), 502
        elif error in ("bot_token_missing", "storage_upload_failed", "firestore_write_failed"):
            return jsonify(result), 500
        else:
            return jsonify(result), 500

    return jsonify(result), 200

@app.post("/slack/sync_batch/start")
def slack_sync_batch_start():
    """
    POST /slack/sync_batch/start
    Body: { uid, agent_id, channels: [{channel_id, channel_name, team_id?}], tz?, limit? }

    Creates a Firestore batch job doc:
      teams/{team_id}/users/{uid}/agents/{agent_id}/slack_sync_batches/{batch_id}

    The batch job holds a queue of channels to sync, plus progress fields.
    """

    payload = request.get_json(silent=True) or {}
    uid = payload.get("uid")
    agent_id = payload.get("agent_id")
    channels = payload.get("channels") or []
    tz = payload.get("tz", "America/New_York")
    limit = int(payload.get("limit", 500))

    if not uid or not agent_id:
        return jsonify({"ok": False, "error": "missing_uid_or_agent_id"}), 400
    if not isinstance(channels, list) or len(channels) == 0:
        return jsonify({"ok": False, "error": "missing_channels"}), 400

    # Validate & normalize channel objects
    normalized = []
    seen = set()
    for c in channels:
        if not isinstance(c, dict):
            continue
        channel_id = (c.get("channel_id") or "").strip()
        channel_name = (c.get("channel_name") or "").strip()
        team_id = (c.get("team_id") or "").strip() or None

        if not channel_id or not channel_name:
            continue

        key = (team_id or "", channel_id)
        if key in seen:
            continue
        seen.add(key)

        normalized.append({
            "channel_id": channel_id,
            "channel_name": channel_name,
            "team_id": team_id,      # optional; get_bot_token_for_uid can also infer
            "status": "queued",      # queued | running | done | error | skipped
            "started_at": None,
            "finished_at": None,
            "result": None,          # store {ok, sync_run_id, storage paths, counts} etc.
            "error": None,
        })

    if not normalized:
        return jsonify({"ok": False, "error": "no_valid_channels"}), 400

    batch_id = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ") + "_" + uuid.uuid4().hex[:8]
    now_unix = int(time.time())

    try:
        team_user_ref, _ = get_team_user_ref(db, uid)
    except KeyError as e:
        return jsonify({"ok": False, "error": str(e)}), 404

    batch_ref = (
        team_user_ref.collection("agents").document(agent_id)
          .collection("slack_sync_batches").document(batch_id)
    )

    # Store queue + progress in one doc (simple for polling)
    # If you anticipate huge batches (100s), switch queue into a subcollection later.
    doc = {
        "batch_id": batch_id,
        "uid": uid,
        "agent_id": agent_id,
        "provider": "slack",
        "status": "running",          # running | done | error | cancelled
        "created_at": firestore.SERVER_TIMESTAMP,
        "started_at_unix": now_unix,
        "finished_at_unix": None,

        "tz": tz,
        "limit": limit,

        "total": len(normalized),
        "completed": 0,
        "failed": 0,

        "queue": normalized,          # list of per-channel work items
        "cursor": 0,                  # next index to process
        "last_tick_at_unix": None,
    }

    try:
        batch_ref.set(doc)
    except Exception as e:
        return jsonify({"ok": False, "error": "firestore_write_failed", "details": str(e)}), 500

    return jsonify({
        "ok": True,
        "batch_id": batch_id,
        "total": len(normalized),
        "status": "running",
        "next": {
            "tick_endpoint": "/slack/sync_batch/tick",
            "status_endpoint": "/slack/sync_batch/status",
        }
    })

@app.get("/slack/remove_connection")
def slack_remove_connection():
    """
    GET /slack/remove_connection?uid=...
    Deletes ALL docs in teams/{team_id}/users/{uid}/slack_installations
    """
    uid = request.args.get("uid")
    if not uid:
        return jsonify({"ok": False, "error": "missing_uid"}), 400

    db = firestore.client()
    try:
        installs_ref, _ = get_slack_installations_ref(db, uid)
    except KeyError as e:
        return jsonify({"ok": False, "error": str(e)}), 404

    snaps = list(installs_ref.stream())
    if not snaps:
        return jsonify({"ok": True, "uid": uid, "deleted_team_ids": [], "deleted_count": 0})

    batch = db.batch()
    deleted_ids = []
    for s in snaps:
        batch.delete(installs_ref.document(s.id))
        deleted_ids.append(s.id)
    batch.commit()

    return jsonify({"ok": True, "uid": uid, "deleted_team_ids": deleted_ids, "deleted_count": len(deleted_ids)})


@app.get("/slack/transcript_threads")
def slack_get_transcript_threads():
    """
    GET /slack/transcript_threads?uid=...&agent_id=...&source_id=...
    Optional: &before_last_message_ts=1767566366.641149
    """

    uid = request.args.get("uid")
    agent_id = request.args.get("agent_id")
    source_id = request.args.get("source_id")  # Slack channel id (your source doc id)
    before_ts = request.args.get("before_last_message_ts")  # optional

    if not uid or not agent_id or not source_id:
        return jsonify({"ok": False, "error": "missing_uid_agent_id_or_source_id"}), 400

    try:
        _, team_id_solari = get_team_user_ref(db, uid)
    except KeyError as e:
        return jsonify({"ok": False, "error": str(e)}), 404

    source_ref = (
        db.collection("teams").document(team_id_solari)
          .collection("agents").document(agent_id)
          .collection("sources").document(source_id)
    )
    if not source_ref.get().exists:
        return jsonify({"ok": False, "error": "source_not_found"}), 404

    chunks_ref = source_ref.collection("transcript_chunks")

    # ---- Query latest chunk by last_message_ts (with optional pagination) ----
    def query_by_last_message_ts():
        q = chunks_ref.order_by("last_message_ts", direction=firestore.Query.DESCENDING)
        if before_ts:
            q = q.where("last_message_ts", "<", before_ts)
        return list(q.limit(1).stream())

    def query_by_created_at_fallback():
        q = chunks_ref.order_by("created_at", direction=firestore.Query.DESCENDING)
        # Pagination fallback: if you want "older" reliably, prefer before_created_at instead.
        # We'll keep it simple: if before_ts is set and last_message_ts ordering fails,
        # we still return the latest by created_at (or you can extend later).
        return list(q.limit(1).stream())

    try:
        chunk_snaps = query_by_last_message_ts()
    except Exception:
        chunk_snaps = query_by_created_at_fallback()

    if not chunk_snaps:
        return jsonify({"ok": False, "error": "no_transcript_chunks_found"}), 404

    chunk = chunk_snaps[0].to_dict() or {}
    storage_path = chunk.get("storage_path_threads")
    if not storage_path:
        return jsonify({"ok": False, "error": "missing_storage_path_threads"}), 500

    # ---- Download threads.json from Firebase Storage ----
    try:
        bucket = storage.bucket()
        blob = bucket.blob(storage_path)
        raw = blob.download_as_text(encoding="utf-8")
        threads_json = json.loads(raw)
    except Exception as e:
        return jsonify({"ok": False, "error": "failed_to_load_threads_json", "details": str(e)}), 500

    # ---- Response (UI-ready) ----
    return jsonify({
        "ok": True,
        "uid": uid,
        "agent_id": agent_id,
        "source_id": source_id,
        "sync_run_id": chunk.get("sync_run_id") or (threads_json.get("sync_run_id") if isinstance(threads_json, dict) else None),
        "created_at": chunk.get("created_at"),
        "last_message_ts": chunk.get("last_message_ts"),
        "storage_path_threads": storage_path,
        "threads_json": threads_json,
    })

@app.get("/slack/transcript_chunks/list")
def slack_list_transcript_chunks():
    uid = request.args.get("uid")
    agent_id = request.args.get("agent_id")
    source_id = request.args.get("source_id")
    limit = int(request.args.get("limit", 20))

    if not uid or not agent_id or not source_id:
        return jsonify({"ok": False, "error": "missing_uid_agent_id_or_source_id"}), 400

    try:
        _, team_id_solari = get_team_user_ref(db, uid)
    except KeyError as e:
        return jsonify({"ok": False, "error": str(e)}), 404

    chunks_ref = (
        db.collection("teams").document(team_id_solari)
          .collection("agents").document(agent_id)
          .collection("sources").document(source_id)
          .collection("transcript_chunks")
    )

    snaps = list(
        chunks_ref.order_by("created_at", direction=firestore.Query.DESCENDING).limit(limit).stream()
    )

    items = []
    for s in snaps:
        d = s.to_dict() or {}
        lmt = d.get("last_message_ts")
        items.append({
            "doc_id": s.id,
            "sync_run_id": d.get("sync_run_id"),
            "created_at": d.get("created_at"),
            "last_message_ts": lmt,
            "last_message_ts_type": type(lmt).__name__,
            "storage_path_threads": d.get("storage_path_threads"),
        })

    return jsonify({"ok": True, "count": len(items), "items": items})

@app.get("/slack/transcript_by_sync_run")
def slack_transcript_by_sync_run():
    """
    GET /slack/transcript_by_sync_run?uid=...&agent_id=...&source_id=...&sync_run_id=...
    Returns the threads.json payload for that exact sync run.
    """

    uid = request.args.get("uid")
    agent_id = request.args.get("agent_id")
    source_id = request.args.get("source_id")  # channel id (your source doc id)
    sync_run_id = request.args.get("sync_run_id")

    if not uid or not agent_id or not source_id or not sync_run_id:
        return jsonify({"ok": False, "error": "missing_uid_agent_id_source_id_or_sync_run_id"}), 400

    try:
        _, team_id_solari = get_team_user_ref(db, uid)
    except KeyError as e:
        return jsonify({"ok": False, "error": str(e)}), 404

    chunk_ref = (
        db.collection("teams").document(team_id_solari)
          .collection("agents").document(agent_id)
          .collection("sources").document(source_id)
          .collection("transcript_chunks").document(sync_run_id)
    )

    snap = chunk_ref.get()
    if not snap.exists:
        return jsonify({"ok": False, "error": "chunk_not_found", "sync_run_id": sync_run_id}), 404

    chunk = snap.to_dict() or {}
    storage_path = chunk.get("storage_path_threads")
    if not storage_path:
        return jsonify({"ok": False, "error": "missing_storage_path_threads", "sync_run_id": sync_run_id}), 500

    # Download the JSON from Firebase Storage
    try:
        bucket = storage.bucket()
        blob = bucket.blob(storage_path)
        raw = blob.download_as_text(encoding="utf-8")
        threads_json = json.loads(raw)
    except Exception as e:
        return jsonify({"ok": False, "error": "failed_to_load_threads_json", "details": str(e)}), 500

    return jsonify({
        "ok": True,
        "uid": uid,
        "agent_id": agent_id,
        "source_id": source_id,
        "sync_run_id": sync_run_id,
        "created_at": chunk.get("created_at"),
        "last_message_ts": chunk.get("last_message_ts"),
        "storage_path_threads": storage_path,  # handy for debugging; remove later if you want
        "threads_json": threads_json,
    })

# endpoint to get the status of a job
@app.get("/slack/sync_batch/status")
def slack_sync_batch_status():
    """
    GET /slack/sync_batch/status?uid=...&agent_id=...&batch_id=...
    """
    uid = request.args.get("uid")
    agent_id = request.args.get("agent_id")
    batch_id = request.args.get("batch_id")

    if not uid or not agent_id or not batch_id:
        return jsonify({"ok": False, "error": "missing_uid_agent_id_or_batch_id"}), 400

    try:
        team_user_ref, _ = get_team_user_ref(db, uid)
    except KeyError as e:
        return jsonify({"ok": False, "error": str(e)}), 404

    batch_ref = (
        team_user_ref.collection("agents").document(agent_id)
          .collection("slack_sync_batches").document(batch_id)
    )

    snap = batch_ref.get()
    if not snap.exists:
        return jsonify({"ok": False, "error": "batch_not_found"}), 404

    doc = snap.to_dict() or {}
    # Return only what the UI needs
    return jsonify({
        "ok": True,
        "batch_id": batch_id,
        "uid": uid,
        "agent_id": agent_id,
        "status": doc.get("status"),
        "total": doc.get("total", 0),
        "completed": doc.get("completed", 0),
        "failed": doc.get("failed", 0),
        "cursor": doc.get("cursor", 0),
        "queue": doc.get("queue", []),
        "last_tick_at_unix": doc.get("last_tick_at_unix"),
        "finished_at_unix": doc.get("finished_at_unix"),
    })

# tick - for the batch job to do an increment of channel syncing work
@app.get("/slack/sync_batch/tick")
def slack_sync_batch_tick():
    """
    GET /slack/sync_batch/tick?uid=...&agent_id=...&batch_id=...
    Processes exactly ONE channel from the queue per call.
    """
    uid = request.args.get("uid")
    agent_id = request.args.get("agent_id")
    batch_id = request.args.get("batch_id")

    if not uid or not agent_id or not batch_id:
        return jsonify({"ok": False, "error": "missing_uid_agent_id_or_batch_id"}), 400

    try:
        team_user_ref, _ = get_team_user_ref(db, uid)
    except KeyError as e:
        return jsonify({"ok": False, "error": str(e)}), 404

    batch_ref = (
        team_user_ref.collection("agents").document(agent_id)
          .collection("slack_sync_batches").document(batch_id)
    )

    now_unix = int(time.time())

    # --- Step A: claim one unit of work (transaction lock) ---
    @firestore.transactional
    def claim_work(txn: firestore.Transaction):
        snap = batch_ref.get(transaction=txn)
        if not snap.exists:
            return {"ok": False, "error": "batch_not_found"}

        doc = snap.to_dict() or {}
        status = doc.get("status")
        if status not in ("running",):
            return {"ok": False, "error": "batch_not_running", "status": status}

        queue = doc.get("queue", []) or []
        cursor = int(doc.get("cursor", 0) or 0)

        if cursor >= len(queue):
            # Mark done
            txn.update(batch_ref, {
                "status": "done",
                "finished_at_unix": doc.get("finished_at_unix") or now_unix,
                "last_tick_at_unix": now_unix,
            })
            return {"ok": True, "done": True, "doc": doc}

        item = queue[cursor] or {}
        # If item is already done/errored for some reason, advance cursor and let next tick handle next
        if item.get("status") in ("done", "error", "skipped"):
            txn.update(batch_ref, {
                "cursor": cursor + 1,
                "last_tick_at_unix": now_unix,
            })
            return {"ok": True, "skipped_cursor_advance": True, "doc": doc}

        # Mark this item as running
        item["status"] = "running"
        item["started_at_unix"] = now_unix

        queue[cursor] = item

        txn.update(batch_ref, {
            "queue": queue,
            "last_tick_at_unix": now_unix,
        })

        # Return the claimed work
        return {
            "ok": True,
            "done": False,
            "cursor": cursor,
            "item": item,
            "doc": doc,
        }

    txn = db.transaction()
    claim = claim_work(txn)

    if not claim.get("ok"):
        return jsonify(claim), (404 if claim.get("error") == "batch_not_found" else 400)

    if claim.get("done"):
        doc = claim.get("doc") or {}
        return jsonify({
            "ok": True,
            "batch_id": batch_id,
            "status": "done",
            "total": doc.get("total", 0),
            "completed": doc.get("completed", 0),
            "failed": doc.get("failed", 0),
        })

    # If we only advanced cursor due to already-done items, tell UI to call again
    if claim.get("skipped_cursor_advance"):
        doc = claim.get("doc") or {}
        return jsonify({
            "ok": True,
            "batch_id": batch_id,
            "status": doc.get("status", "running"),
            "total": doc.get("total", 0),
            "completed": doc.get("completed", 0),
            "failed": doc.get("failed", 0),
            "note": "advanced_cursor_over_completed_item",
        })

    cursor = claim["cursor"]
    item = claim["item"]
    tz = (claim.get("doc") or {}).get("tz", "America/New_York")
    limit = int((claim.get("doc") or {}).get("limit", 500))

    channel_id = item.get("channel_id")
    channel_name = item.get("channel_name")
    team_id = item.get("team_id")  # optional

    # --- Step B: do the actual channel sync (this is where transcripts are generated) ---
    result = _sync_channel_transcript_internal(
        uid=uid,
        agent_id=agent_id,
        channel_id=channel_id,
        channel_name=channel_name,
        team_id=team_id,
        tz=tz,
        limit=limit,
    )

    # --- Step C: write result back & advance cursor ---
    @firestore.transactional
    def commit_result(txn: firestore.Transaction):
        snap = batch_ref.get(transaction=txn)
        if not snap.exists:
            return {"ok": False, "error": "batch_not_found"}

        doc = snap.to_dict() or {}
        queue = doc.get("queue", []) or []
        total = int(doc.get("total", len(queue)) or len(queue))
        completed = int(doc.get("completed", 0) or 0)
        failed = int(doc.get("failed", 0) or 0)

        if cursor >= len(queue):
            return {"ok": False, "error": "cursor_out_of_bounds"}

        qitem = queue[cursor] or {}
        qitem["finished_at_unix"] = now_unix
        qitem["result"] = result if isinstance(result, dict) else {"ok": False, "error": "bad_result_shape"}

        if result.get("ok"):
            qitem["status"] = "done"
            completed += 1
        else:
            qitem["status"] = "error"
            qitem["error"] = result.get("error") or "unknown_error"
            failed += 1

        queue[cursor] = qitem

        new_cursor = cursor + 1
        new_status = "running"
        finished_at_unix = None
        if new_cursor >= len(queue):
            new_status = "done"
            finished_at_unix = now_unix

        txn.update(batch_ref, {
            "queue": queue,
            "cursor": new_cursor,
            "completed": completed,
            "failed": failed,
            "status": new_status,
            "finished_at_unix": finished_at_unix,
            "last_tick_at_unix": now_unix,
        })

        return {
            "ok": True,
            "status": new_status,
            "total": total,
            "completed": completed,
            "failed": failed,
            "just_processed": {
                "channel_id": channel_id,
                "channel_name": channel_name,
                "result": result,
            }
        }

    txn2 = db.transaction()
    out = commit_result(txn2)

    # Return UI-friendly progress
    return jsonify({
        "ok": True,
        "batch_id": batch_id,
        **out
    })

# retry a channel + clean up the batch tracking in firebase
@app.route("/slack/batch_channel_retry", methods=["POST"])
def slack_batch_channel_retry():
    """
    POST /slack/batch_channel_retry?uid=...&agent_id=...&channel_id=...&channel_name=...&batch_id=...
    Optional: &team_id=...&tz=America/New_York&limit=500
    
    Syncs a channel and updates the batch queue to mark that channel as done.
    """
    uid = request.args.get("uid")
    agent_id = request.args.get("agent_id")
    channel_id = request.args.get("channel_id")
    channel_name = request.args.get("channel_name")
    batch_id = request.args.get("batch_id")
    team_id = request.args.get("team_id")
    tz = request.args.get("tz", "America/New_York")
    max_messages = int(request.args.get("limit", 500))

    if not uid or not agent_id or not channel_id or not channel_name or not batch_id:
        return jsonify({
            "ok": False,
            "error": "missing_uid_agent_id_channel_id_channel_name_or_batch_id"
        }), 400

    # Step 1: Sync the channel
    result = _sync_channel_transcript_internal(
        uid=uid,
        agent_id=agent_id,
        channel_id=channel_id,
        channel_name=channel_name,
        team_id=team_id,
        tz=tz,
        limit=max_messages,
    )

    # Step 2 & 3: Update the batch queue
    try:
        team_user_ref, _ = get_team_user_ref(db, uid)
    except KeyError as e:
        return jsonify({"ok": False, "error": str(e)}), 404

    batch_ref = (
        team_user_ref.collection("agents").document(agent_id)
          .collection("slack_sync_batches").document(batch_id)
    )

    now_unix = int(time.time())

    @firestore.transactional
    def update_batch_queue(txn: firestore.Transaction):
        snap = batch_ref.get(transaction=txn)
        if not snap.exists:
            return {"ok": False, "error": "batch_not_found"}

        doc = snap.to_dict() or {}
        queue = doc.get("queue", []) or []
        completed = int(doc.get("completed", 0) or 0)
        failed = int(doc.get("failed", 0) or 0)

        # Find the channel in the queue
        channel_found = False
        for i, item in enumerate(queue):
            item_channel_id = item.get("channel_id")
            item_team_id = item.get("team_id")
            
            # Match by channel_id, and optionally by team_id if both are present
            if item_channel_id == channel_id:
                if team_id is None or item_team_id == team_id or (item_team_id is None and team_id is None):
                    # Found the channel, update it with latest result
                    previous_status = item.get("status", "queued")
                    
                    queue[i]["finished_at_unix"] = now_unix
                    queue[i]["result"] = result if isinstance(result, dict) else {"ok": False, "error": "bad_result_shape"}
                    
                    if result.get("ok"):
                        # Success: mark as done
                        queue[i]["status"] = "done"
                        queue[i].pop("error", None)  # Remove error field if it exists
                        
                        # Adjust counts: if it was previously "error", move from failed to completed
                        if previous_status == "error":
                            failed = max(0, failed - 1)  # Don't go negative
                            completed += 1
                        elif previous_status != "done":
                            # Was queued/running, now done
                            completed += 1
                    else:
                        # Error: mark as error with new error message
                        queue[i]["status"] = "error"
                        queue[i]["error"] = result.get("error") or "unknown_error"
                        
                        # Adjust counts: if it was previously "done", move from completed to failed
                        if previous_status == "done":
                            completed = max(0, completed - 1)  # Don't go negative
                            failed += 1
                        elif previous_status != "error":
                            # Was queued/running, now error
                            failed += 1
                    
                    channel_found = True
                    break

        if not channel_found:
            return {"ok": False, "error": "channel_not_found_in_batch_queue"}

        # Update the batch document
        txn.update(batch_ref, {
            "queue": queue,
            "completed": completed,
            "failed": failed,
            "last_tick_at_unix": now_unix,
        })

        return {
            "ok": True,
            "completed": completed,
            "failed": failed,
            "total": doc.get("total", len(queue)),
        }

    txn = db.transaction()
    batch_update = update_batch_queue(txn)

    if not batch_update.get("ok"):
        error = batch_update.get("error", "")
        if error == "batch_not_found":
            return jsonify(batch_update), 404
        elif error == "channel_not_found_in_batch_queue":
            return jsonify(batch_update), 400
        else:
            return jsonify(batch_update), 500

    # Return combined result
    return jsonify({
        "ok": True,
        "sync_result": result,
        "batch_id": batch_id,
        "completed": batch_update.get("completed"),
        "failed": batch_update.get("failed"),
        "total": batch_update.get("total"),
    }), 200

def _new_6digit_code() -> str:
    return str(secrets.randbelow(1_000_000)).zfill(6)

@app.post("/teams/create_invite_code")
def teams_create_invite_code():
    """
    POST /teams/create_invite_code
    Body JSON:
      { "uid": "...", "team_id": "..." }

    Claims a globally-unique 6-digit code by creating:
      team_invite_codes/{code} -> { teamId, created_by_uid, created_at }
    """
    body = request.get_json(silent=True) or {}
    uid = (body.get("uid") or "").strip()
    team_id = (body.get("team_id") or "").strip()

    db = firestore.client()

    if not uid or not team_id:
        return jsonify({"ok": False, "error": "missing_uid_or_team_id"}), 400

    # Optional: verify team exists
    team_ref = db.collection("teams").document(team_id)
    if not team_ref.get().exists:
        return jsonify({"ok": False, "error": "team_not_found"}), 404

    max_attempts = 50
    for _ in range(max_attempts):
        code = _new_6digit_code()
        code_ref = db.collection("team_invite_codes").document(code)

        try:
            # create() is atomic and fails if doc already exists
            code_ref.create({
                "teamId": team_id,
                "created_by_uid": uid,
                "created_at": firestore.SERVER_TIMESTAMP,
                "status": "active",
                "code": code,
            })

            # NEW: persist invite code on team doc
            team_ref.set({
                "invite_code": code,
            }, merge=True)

            return jsonify({"ok": True, "team_id": team_id, "invite_code": code})
        except Exception as e:
            msg = str(e).lower()
            if "already exists" in msg or "alreadyexists" in msg:
                continue
            return jsonify({"ok": False, "error": "firestore_error", "details": str(e)}), 500

    return jsonify({"ok": False, "error": "could_not_allocate_unique_code"}), 503

@app.post("/team/join_team_invite_code")
def team_join_team_invite_code():
    """
    POST /team/join_team_invite_code
    Body JSON:
      { "invite_code": "123456" }

    Returns the team_id for a valid invite code.
    """
    body = request.get_json(silent=True) or {}
    invite_code = (body.get("invite_code") or "").strip()

    if not invite_code:
        return jsonify({"ok": False, "error": "missing_invite_code"}), 400

    db = firestore.client()
    code_ref = db.collection("team_invite_codes").document(invite_code)
    snap = code_ref.get()

    if not snap.exists:
        return jsonify({"ok": False, "error": "invite_code_not_found"}), 404

    data = snap.to_dict() or {}
    # Accept legacy or new field names for compatibility.
    team_id = data.get("team_id") or data.get("teamId")
    if not team_id:
        return jsonify({"ok": False, "error": "team_id_missing_for_code"}), 500

    return jsonify({"ok": True, "invite_code": invite_code, "team_id": team_id})

# @app.post("/teams/invite_members")
# def teams_invite_members():
#     """
#     POST /teams/invite_members
#     Body JSON:
#       { "team_id": "...", "emails": ["a@b.com", "c@d.com"] }

#     For now, just log the payload and return a confirmation response.
#     """
#     body = request.get_json(silent=True) or {}
#     team_id = (body.get("team_id") or "").strip()
#     emails = body.get("emails")

#     if not team_id:
#         return jsonify({"ok": False, "error": "missing_team_id"}), 400
#     if not isinstance(emails, list) or not emails:
#         return jsonify({"ok": False, "error": "missing_emails"}), 400

#     cleaned_emails = [str(e).strip() for e in emails if str(e).strip()]
#     if not cleaned_emails:
#         return jsonify({"ok": False, "error": "no_valid_emails"}), 400

#     logger.info(f"Invite members requested for team_id={team_id}, emails={cleaned_emails}")
#     return jsonify({"ok": True, "team_id": team_id, "emails": cleaned_emails})

@app.post("/teams/get_creator_first_name")
def teams_get_creator_first_name():
    """
    POST /teams/get_creator_first_name
    Body JSON:
      { "team_id": "...", "user_id": "..." }

    Returns the first name of the user who created the team, scoped to a user in that team.
    """
    body = request.get_json(silent=True) or {}
    team_id = (body.get("team_id") or "").strip()
    user_id = (body.get("user_id") or "").strip()

    first_name, creator_uid, error = _get_team_creator_first_name(team_id, user_id)
    if error:
        return jsonify({"ok": False, "error": error["message"]}), error["status"]

    return jsonify({
        "ok": True,
        "team_id": team_id,
        "creator_uid": str(creator_uid),
        "creator_first_name": first_name,
    })

@app.post("/api/team/list_members")
@require_solari_key
def team_list_members():
    """
    POST /api/team/list_members
    Body JSON:
      { "teamId": "...", "userId": "..." }

    Returns displayName, email, and role for each team member.
    """
    body = request.get_json(silent=True) or {}
    team_id = (body.get("teamId") or "").strip()
    user_id = (body.get("userId") or "").strip()

    if not team_id or not user_id:
        return jsonify({"ok": False, "error": "missing_team_id_or_user_id"}), 400

    db = firestore.client()
    user_snap = db.collection("users").document(user_id).get()
    if not user_snap.exists:
        return jsonify({"ok": False, "error": "user_not_found"}), 404

    user_data = user_snap.to_dict() or {}
    if (user_data.get("teamId") or "").strip() != team_id:
        return jsonify({"ok": False, "error": "user_not_in_team"}), 403

    members_ref = db.collection("teams").document(team_id).collection("users")
    members_snaps = members_ref.stream()

    members = []
    for snap in members_snaps:
        data = snap.to_dict() or {}
        members.append({
            "uid": data.get("uid") or snap.id,
            "displayName": data.get("displayName"),
            "email": data.get("email"),
            "role": data.get("role"),
            "agents": data.get("agents") or [],
        })

    return jsonify({
        "ok": True,
        "team_id": team_id,
        "members": members,
    })

@app.post("/api/team/update-member-role")
@require_solari_key
def team_update_member_role():
    """
    POST /api/team/update-member-role
    Body JSON:
      { "teamId": "...", "userId": "...", "role": "..." }
    """
    body = request.get_json(silent=True) or {}
    team_id = (body.get("teamId") or "").strip()
    user_id = (body.get("userId") or "").strip()
    role = (body.get("role") or "").strip()

    if not team_id or not user_id or not role:
        return jsonify({"ok": False, "error": "missing_team_id_user_id_or_role"}), 400

    db = firestore.client()
    user_snap = db.collection("users").document(user_id).get()
    if not user_snap.exists:
        return jsonify({"ok": False, "error": "user_not_found"}), 404

    user_data = user_snap.to_dict() or {}
    if (user_data.get("teamId") or "").strip() != team_id:
        return jsonify({"ok": False, "error": "user_not_in_team"}), 403

    member_ref = db.collection("teams").document(team_id).collection("users").document(user_id)
    member_snap = member_ref.get()
    if not member_snap.exists:
        return jsonify({"ok": False, "error": "member_not_found"}), 404

    member_ref.set({"role": role}, merge=True)

    return jsonify({
        "ok": True,
        "team_id": team_id,
        "user_id": user_id,
        "role": role,
    })

def _get_team_creator_first_name(team_id: str, user_id: str):
    if not team_id or not user_id:
        return None, None, {"message": "missing_team_id_or_user_id", "status": 400}

    db = firestore.client()
    team_ref = db.collection("teams").document(team_id)
    team_snap = team_ref.get()
    if not team_snap.exists:
        return None, None, {"message": "team_not_found", "status": 404}

    user_ref = db.collection("users").document(str(user_id))
    user_snap = user_ref.get()
    if not user_snap.exists:
        return None, None, {"message": "user_not_found", "status": 404}

    user_data = user_snap.to_dict() or {}
    if (user_data.get("teamId") or "").strip() != team_id:
        return None, None, {"message": "user_not_in_team", "status": 403}

    team_data = team_snap.to_dict() or {}
    creator_uid = team_data.get("createdBy")
    if not creator_uid:
        return None, None, {"message": "team_creator_missing", "status": 500}

    creator_ref = db.collection("users").document(str(creator_uid))
    creator_snap = creator_ref.get()
    if not creator_snap.exists:
        return None, None, {"message": "team_creator_not_found", "status": 404}

    creator_data = creator_snap.to_dict() or {}
    display_name = (creator_data.get("displayName") or "").strip()
    if not display_name:
        return None, None, {"message": "creator_display_name_missing", "status": 500}

    first_name = display_name.split()[0]
    return first_name, creator_uid, None

if __name__ == '__main__':
    # Get port from environment or default to 5000
    port = int(get_env_var('PORT', default='5000'))
    debug = get_env_var('FLASK_DEBUG', default='False').lower() == 'true'
    
    app.run(host='0.0.0.0', port=port, debug=debug)

