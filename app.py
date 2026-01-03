import os
import json
import io
import tempfile
import time
from datetime import datetime, timedelta
from typing import Optional
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
ATLASSIAN_REDIRECT_URI = 'https://solari-backend.onrender.com/auth/jira/callback'
FRONTEND_SUCCESS_URL = os.getenv('FRONTEND_SUCCESS_URL', 'http://localhost:3000')

# Security: Solari Key validation
SOLARI_INTERNAL_KEY = os.environ.get("SOLARI_INTERNAL_KEY")
SOLARI_DEV_KEY = os.environ.get("SOLARI_DEV_KEY")
SOLARI_KEY_HEADER = "x-solari-key"

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
    api_key = get_api_key('OPENAI_API_KEY')
    return OpenAI(api_key=api_key)

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
    client_id = request.args.get("client_id")
    
    if not uid:
        return jsonify({"error": "Missing uid parameter"}), 400
    
    if not client_id:
        return jsonify({"error": "Missing client_id parameter"}), 400
    
    query_params = {
        "audience": "api.atlassian.com",
        "client_id": client_id,
        "scope": "read:jira-work offline_access",
        "redirect_uri": ATLASSIAN_REDIRECT_URI,
        "response_type": "code",
        "prompt": "consent",
        "state": uid  # ✅ safely carries the user ID through OAuth
    }
    
    auth_url = "https://auth.atlassian.com/authorize?" + urlencode(query_params)
    return redirect(auth_url)

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

        # 4. ✅ Save tokens securely to Firestore
        try:
            db = firestore.client()
            db.collection("users").document(uid).set({
                "jira_access_token": access_token,
                "jira_refresh_token": refresh_token,
                "jira_cloud_id": cloud_id,
                "jira_site_url": jira_site_url,
                "jira_connected": True,
                "jira_connected_at": firestore.SERVER_TIMESTAMP,
                "jira_expires_at": datetime.utcnow() + timedelta(seconds=expires_in)
            }, merge=True)
            logger.info(f"✅ Jira OAuth saved in Firestore for user {uid}")
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

def _get_jira_creds(user_id: str):
    """
    Retrieve Jira OAuth credentials from Firestore for a user.
    
    Returns:
        dict with keys: access_token, cloud_id, site_url
    Raises:
        Exception if user not found or credentials missing
    """
    db = firestore.client()
    user_doc = db.collection("users").document(user_id).get()
    
    if not user_doc.exists:
        raise Exception(f"User {user_id} not found in Firestore")
    
    user_data = user_doc.to_dict() or {}
    access_token = user_data.get("jira_access_token")
    cloud_id = user_data.get("jira_cloud_id")
    
    if not access_token or not cloud_id:
        raise Exception(f"Jira credentials not found for user {user_id}. Please complete OAuth flow.")
    
    return {
        "access_token": access_token,
        "cloud_id": cloud_id,
        "site_url": user_data.get("jira_site_url")
    }

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

@app.route('/api/pinecone_doc_upload', methods=['POST'])
@require_solari_key
def pinecone_doc_upload():
    """
    Upload documents to Pinecone after processing.
    
    Expected request body:
    {
        "namespace": "your-namespace",
        "file_path": "users/userId/files/file.pdf",
        "nickname": "optional-nickname"  # optional
    }
    
    Process:
    1. Download file from Firebase Storage
    2. Extract text from file (PDF/DOCX)
    3. Chunk the text
    4. Generate embeddings using OpenAI
    5. Upload to Pinecone
    
    Returns:
        JSON response with upload status
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
        namespace = data.get('namespace')
        if not namespace:
            return jsonify({
                'status': 'error',
                'message': 'namespace parameter is required'
            }), 400
        
        file_path = data.get('file_path')
        if not file_path:
            return jsonify({
                'status': 'error',
                'message': 'file_path parameter is required'
            }), 400
        
        # Optional parameter
        nickname = data.get('nickname', '')
        
        logger.info(f"Processing file: {file_path} for namespace: {namespace}")
        
        # Step 1: Download file from Firebase Storage
        logger.info("Downloading file from Firebase Storage...")
        file_content = download_file_from_firebase(file_path)
        
        # Step 2: Extract text from file
        logger.info("Extracting text from file...")
        text = extract_text_from_file(file_content, file_path)
        
        if not text or len(text.strip()) == 0:
            return jsonify({
                'status': 'error',
                'message': 'No text could be extracted from the file'
            }), 400
        
        logger.info(f"Extracted {len(text)} characters of text")
        
        # Step 3: Chunk the text
        logger.info("Chunking text...")
        text_chunks = chunk_text(text, chunk_size=1000, chunk_overlap=200)
        logger.info(f"Created {len(text_chunks)} chunks")
        
        # Step 4: Generate embeddings
        logger.info("Generating embeddings...")
        openai_client = get_openai_client()
        embeddings = generate_embeddings(text_chunks, openai_client)
        
        # Step 5: Prepare vectors for Pinecone
        # Use file_path as base for IDs (sanitize it)
        file_id_base = file_path.replace('/', '_').replace(' ', '_')
        vectors = []
        for i, (chunk, embedding) in enumerate(zip(text_chunks, embeddings)):
            vectors.append({
                'id': f"{file_id_base}_chunk_{i}",
                'values': embedding,
                'metadata': {
                    'file_path': file_path,
                    'chunk_index': i,
                    'text_preview': chunk[:500],  # Store first 500 chars as metadata
                    'nickname': nickname  # Add nickname to metadata
                }
            })
        
        # Step 6: Upload to Pinecone in batches
        logger.info(f"Uploading {len(vectors)} vectors to Pinecone in batches...")
        index_name = 'production'
        total_uploaded = upload_vectors_to_pinecone(vectors, namespace, index_name=index_name, batch_size=100)
        
        logger.info(f"Successfully uploaded {total_uploaded} vectors to namespace '{namespace}'")
        
        return jsonify({
            'status': 'success',
            'message': f'Successfully processed and uploaded {len(vectors)} vectors to namespace "{namespace}"',
            'namespace': namespace,
            'index': index_name,
            'file_path': file_path,
            'vectors_uploaded': len(vectors),
            'chunks_created': len(text_chunks),
            'text_length': len(text),
            'nickname': nickname
        }), 200
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'Failed to process file: {str(e)}'
        }), 500

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

# Add this endpoint after the pinecone_doc_upload endpoint (around line 544)

@app.route('/api/pinecone_website_upload', methods=['POST'])
@require_solari_key
def pinecone_website_upload():
    """
    Upload website content to Pinecone after scraping and processing.
    
    Expected request body:
    {
        "namespace": "your-namespace",
        "url": "https://example.com",
        "nickname": "optional-nickname",  # optional
        "chunk_size": 1000,  # optional, default 1000
        "chunk_overlap": 200  # optional, default 200
    }
    
    Process:
    1. Scrape website URL with Firecrawl to get markdown
    2. Chunk the markdown
    3. Generate embeddings using OpenAI
    4. Upload to Pinecone
    
    Returns:
        JSON response with upload status
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
        namespace = data.get('namespace')
        if not namespace:
            return jsonify({
                'status': 'error',
                'message': 'namespace parameter is required'
            }), 400
        
        url = data.get('url')
        if not url:
            return jsonify({
                'status': 'error',
                'message': 'url parameter is required'
            }), 400
        
        # Optional parameters with defaults
        chunk_size = data.get('chunk_size', 1000)
        chunk_overlap = data.get('chunk_overlap', 200)
        nickname = data.get('nickname', '')
        
        logger.info(f"Processing website: {url} for namespace: {namespace}")
        
        # Step 1: Scrape website with Firecrawl
        logger.info("Scraping website with Firecrawl...")
        scrape_result = scrape_website_with_firecrawl(url)
        markdown = scrape_result["markdown"]
        website_metadata = scrape_result["metadata"]
        
        if not markdown or len(markdown.strip()) == 0:
            return jsonify({
                'status': 'error',
                'message': 'No content could be extracted from the website'
            }), 400
        
        logger.info(f"Extracted {len(markdown)} characters of markdown")
        
        # Step 2: Chunk the markdown
        logger.info("Chunking markdown...")
        text_chunks = chunk_text(markdown, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        logger.info(f"Created {len(text_chunks)} chunks")
        
        # Step 3: Generate embeddings
        logger.info("Generating embeddings...")
        openai_client = get_openai_client()
        embeddings = generate_embeddings(text_chunks, openai_client)
        
        # Step 4: Prepare vectors for Pinecone
        # Sanitize URL for use in IDs
        url_id_base = url.replace('https://', '').replace('http://', '').replace('/', '_').replace(' ', '_').replace('.', '_')
        # Remove any remaining special characters that might cause issues
        url_id_base = ''.join(c if c.isalnum() or c == '_' else '_' for c in url_id_base)
        
        vectors = []
        for i, (chunk, embedding) in enumerate(zip(text_chunks, embeddings)):
            vectors.append({
                'id': f"{url_id_base}_chunk_{i}",
                'values': embedding,
                'metadata': {
                    'url': url,
                    'chunk_index': i,
                    'text_preview': chunk[:500],  # Store first 500 chars as metadata
                    'source': 'website',
                    'nickname': nickname,  # Add nickname to metadata
                    # Include relevant metadata from Firecrawl
                    'title': website_metadata.get('title', ''),
                    'description': website_metadata.get('description', ''),
                    'sourceURL': website_metadata.get('sourceURL', url),
                }
            })
        
        # Step 5: Upload to Pinecone in batches
        logger.info(f"Uploading {len(vectors)} vectors to Pinecone in batches...")
        index_name = 'production'
        total_uploaded = upload_vectors_to_pinecone(vectors, namespace, index_name=index_name, batch_size=100)
        
        logger.info(f"Successfully uploaded {total_uploaded} vectors to namespace '{namespace}'")
        
        return jsonify({
            'status': 'success',
            'message': f'Successfully processed and uploaded {len(vectors)} vectors to namespace "{namespace}"',
            'namespace': namespace,
            'index': index_name,
            'url': url,
            'vectors_uploaded': len(vectors),
            'chunks_created': len(text_chunks),
            'markdown_length': len(markdown),
            'website_metadata': website_metadata,
            'nickname': nickname
        }), 200
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400
    except Exception as e:
        logger.error(f"Error processing website: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'Failed to process website: {str(e)}'
        }), 500

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

def download_table_from_source(user_id: str, agent_id: str, document_id: str) -> dict:
    """
    Download a table file (CSV, Excel, etc.) from Firebase Storage using a Firestore source document
    and return both the file content and metadata.
    
    Args:
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
        logger.info(f"Downloading table for user: {user_id}, agent: {agent_id}, document: {document_id}")
        
        # Step 1: Get document from Firestore
        db = firestore.client()
        doc_ref = db.collection('users').document(user_id).collection('agents').document(agent_id).collection('sources').document(document_id)
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

@app.route('/api/table/analyze', methods=['POST'])
@require_solari_key
def analyze_table():
    """
    Analyze a tabular data file from a Firestore document.
    
    Expected request body:
    {
        "user_id": "jyh2RyS8Mvb9OCWF7pKRKEAGxZP2",
        "agent_id": "agent123",
        "document_id": "miRtr9IDqCzu66rBksTG"
    }
    
    Process:
    1. Get document from Firestore at users/{user_id}/agents/{agent_id}/sources/{document_id}
    2. Extract filePath from document
    3. Download file from Firebase Storage
    4. Analyze file to get row count and column types
    5. Return results
    
    Returns:
        JSON response with row count and column information
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
        
        logger.info(f"Analyzing table for user: {user_id}, agent: {agent_id}, document: {document_id}")
        
        # Step 1: Get document from Firestore
        db = firestore.client()
        doc_ref = db.collection('users').document(user_id).collection('agents').document(agent_id).collection('sources').document(document_id)
        doc = doc_ref.get()
        
        if not doc.exists:
            return jsonify({
                'status': 'error',
                'message': f'Document not found: {document_id}'
            }), 404
        
        doc_data = doc.to_dict()
        
        # Step 2: Extract filePath
        file_path = doc_data.get('filePath')
        if not file_path:
            return jsonify({
                'status': 'error',
                'message': 'filePath not found in document'
            }), 400
        
        logger.info(f"Found filePath: {file_path}")
        
        # Step 3: Download file from Firebase Storage
        logger.info("Downloading file from Firebase Storage...")
        file_content = download_file_from_firebase(file_path)
        
        # Step 4: Analyze tabular data
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
        
        return jsonify({
            'status': 'success',
            'message': 'Successfully analyzed tabular data',
            'document_id': document_id,
            'file_path': file_path,
            'row_count': analysis_result['row_count'],
            'columns': analysis_result['columns'],
            'column_count': len(analysis_result['columns'])
        }), 200
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400
    except Exception as e:
        logger.error(f"Error analyzing table: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'Failed to analyze table: {str(e)}'
        }), 500

@app.route('/api/table/download', methods=['POST'])
@require_solari_key
def download_table():
    """
    Test endpoint to download a table file from Firebase Storage using a Firestore source document.
    
    Expected request body:
    {
        "user_id": "jyh2RyS8Mvb9OCWF7pKRKEAGxZP2",
        "agent_id": "agent123",
        "document_id": "MV9pGL1YP6iLdCeBxKay"
    }
    
    Returns:
        JSON response with file metadata (row count, columns, types) and file size.
        Note: file_content is bytes and not included in JSON response, but file_size is provided.
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
        
        logger.info(f"Testing download_table_from_source for user: {user_id}, agent: {agent_id}, document: {document_id}")
        
        # Call the function
        result = download_table_from_source(user_id, agent_id, document_id)
        
        # Prepare response (exclude file_content bytes, but include file size)
        response_data = {
            'status': 'success',
            'message': 'Successfully downloaded and analyzed table',
            'document_id': document_id,
            'file_path': result['file_path'],
            'file_size_bytes': len(result['file_content']),
            'row_count': result['row_count'],
            'columns': result['columns'],
            'column_count': result['column_count']
        }
        
        logger.info(f"Test successful: {result['row_count']} rows, {result['column_count']} columns, {len(result['file_content'])} bytes")
        
        return jsonify(response_data), 200
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400
    except Exception as e:
        logger.error(f"Error downloading table: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'Failed to download table: {str(e)}'
        }), 500

@app.route('/api/table/prepare', methods=['POST'])
@require_solari_key
def prepare_table():
    """
    Download table, extract metadata, and save to temp folder for DuckDB processing.
    
    Expected request body:
    {
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
        
        logger.info(f"Preparing table for DuckDB: user: {user_id}, agent: {agent_id}, document: {document_id}")
        
        # Step 1: Download file and get metadata
        logger.info("Step 1: Downloading table and extracting metadata...")
        download_result = download_table_from_source(user_id, agent_id, document_id)
        
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
        
        logger.info(f"Querying table: user: {user_id}, agent: {agent_id}, document: {document_id}, limit: {limit}")
        
        # Step 1: Download file and get metadata
        logger.info("Step 1: Downloading table and extracting metadata...")
        download_result = download_table_from_source(user_id, agent_id, document_id)
        
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

def get_table_meta(user_id: str, document_id: str, agent_id: str = None) -> Dict[str, Any]:
    """
    Get table metadata from Firestore source document.
    
    Args:
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
    
    doc_ref = db.collection('users').document(user_id).collection('agents').document(agent_id).collection('sources').document(document_id)
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
        logger.info(f"Getting table metadata for user: {user_id}, agent_id: {agent_id}, source_id: {source_id}")
        table_meta = get_table_meta(user_id, source_id, agent_id=agent_id)
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
        logger.info(f"Getting table metadata for user: {user_id}, agent_id: {agent_id}, source_id: {source_id}")
        table_meta = get_table_meta(user_id, source_id, agent_id=agent_id)
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
    user_id: str,
    document_id: str,
    agent_id: str,
    llm_output: Dict[str, Any],
    get_table_meta_fn,
) -> Dict[str, Any]:
    """
    get_table_meta_fn should be your function:
      get_table_meta(user_id, document_id, agent_id) -> dict (table_meta)

    llm_output is what you pasted:
      {"plan": {...}, "planner_schema": [...], ...}

    Returns a dict with ok/issues and debug info.
    """
    table_meta = get_table_meta_fn(user_id, document_id, agent_id)

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

def download_table_csv_to_temp(user_id: str, source_id: str, agent_id: str) -> str:
    """
    Download table file and save to temp directory as CSV.
    Returns the local file path.
    Reuses download_table_from_source and materialize_table_file_atomic.
    """
    download_result = download_table_from_source(user_id, agent_id, source_id)
    
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

    user_id = body.get("user_id")
    source_id = body.get("source_id")   # Firestore doc id for the table
    agent_id = body.get("agent_id")
    raw_plan = body.get("plan")

    if not user_id or not source_id or not agent_id or not isinstance(raw_plan, dict):
        return jsonify({
            "success": False,
            "error": "Required: user_id, source_id, agent_id, plan (object)"
        }), 400

    try:
        # 1) Read schema/meta from Firestore
        table_meta = get_table_meta(user_id, source_id, agent_id)
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
        local_csv_path = download_table_csv_to_temp(user_id, source_id, agent_id)
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

def query_table_endpoint_internal(user_id: str, source_id: str, agent_id: str, question: str):
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
        table_meta = get_table_meta(user_id, source_id, agent_id)
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
            }), 200

        final_plan = vf["final_plan"]

        # 4) Download CSV to temp + run DuckDB
        local_csv_path = download_table_csv_to_temp(user_id, source_id, agent_id)
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
        }), 200

    except Exception as e:
        logger.error(f"Error querying table: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e),
            "ms": int((time.time() - t0) * 1000),
        }), 500

@app.route("/api/table/ask/ai_sql", methods=["POST"])
@require_solari_key
def query_table_endpoint():
    body = request.get_json(force=True) or {}

    user_id = body.get("user_id")
    source_id = body.get("source_id")
    agent_id = body.get("agent_id")
    question = body.get("question")

    if not user_id or not source_id or not agent_id or not question:
        return jsonify({
            "success": False,
            "error": "Required: user_id, source_id, agent_id, question"
        }), 400

    return query_table_endpoint_internal(user_id, source_id, agent_id, question)

@app.route("/api/table/plan/check", methods=["POST"])
@require_solari_key
def check_table_plan():
    """
    Sanity check a table plan against Firestore metadata.
    
    Expected request body:
    {
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
    user_id = body.get("user_id")
    document_id = body.get("document_id")
    agent_id = body.get("agent_id")

    if not user_id or not document_id or not agent_id:
        return jsonify({
            "success": False,
            "error": "Missing required fields: user_id, document_id, agent_id"
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

def perform_rag_query(userid: str, namespace: str, query: str, nickname: str = '', source_type: str = ''):
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
        logger.info("Generating answer using OpenAI...")
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        
        answer = response.choices[0].message.content
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

def select_source_with_openai(query: str, sources: list, model: str = "gpt-3.5-turbo", temperature: float = 0.3) -> str:
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
    openai_client = get_openai_client()
    
    # Build prompts
    prompts = build_source_selection_prompt(query, sources)
    
    logger.info("Using OpenAI to determine best source...")
    response = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompts['system_prompt']},
            {"role": "user", "content": prompts['user_prompt']}
        ],
        temperature=temperature
    )
    
    selected_nickname = response.choices[0].message.content.strip()
    logger.info(f"OpenAI selected nickname: {selected_nickname}")
    
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

def decide_source(userid: str, agent_id: str, namespace: str, query: str, source_type: str = ''):
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
        logger.info(f"Deciding source for user: {userid}, agent: {agent_id}, namespace: {namespace}, query: {query[:100]}...")
        
        # Step 1: Get list of sources from Firestore
        db = firestore.client()
        sources_ref = db.collection('users').document(userid).collection('agents').document(agent_id).collection('sources')
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
        selected_nickname = select_source_with_openai(query, sources)
        
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
        
        # Validate required fields
        userid = data.get('userid')
        if not userid:
            return jsonify({
                'success': False,
                'answer': None,
                'metadata': None,
                'error': 'userid parameter is required'
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
                'error': 'query parameter is required'
            }), 400
        
        # Optional parameters
        nickname = data.get('nickname', '')
        source_type = data.get('source_type', '')
        
        # Perform RAG query
        result = perform_rag_query(userid, namespace, query, nickname, source_type)
        
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
        
        # Validate required fields
        userid = data.get('userid')
        if not userid:
            return jsonify({
                'success': False,
                'answer': None,
                'metadata': None,
                'error': 'userid parameter is required'
            }), 400
        
        query = data.get('query')
        if not query:
            return jsonify({
                'success': False,
                'answer': None,
                'metadata': None,
                'error': 'query parameter is required'
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
                'error': 'agent_id parameter is required when nickname is provided'
            }), 400
        
        logger.info(f"Source confirmed for user: {userid}, agent: {agent_id}, namespace: {namespace}, nickname: {nickname}, query: {query[:100]}...")
        
        # If nickname is provided, check source type and route accordingly
        if nickname and agent_id:
            try:
                db = firestore.client()
                sources_ref = db.collection('users').document(userid).collection('agents').document(agent_id).collection('sources')
                
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
                        return query_table_endpoint_internal(userid, source_id, agent_id, query)
                    else:
                        # Not a table, proceed with RAG query (namespace required)
                        if not namespace:
                            return jsonify({
                                'success': False,
                                'answer': None,
                                'metadata': None,
                                'error': 'namespace parameter is required for RAG queries'
                            }), 400
                        
                        logger.info(f"Source is not a table, running RAG query...")
                        result = perform_rag_query(userid, namespace, query, nickname, source_type)
                        
                        if result['success']:
                            return jsonify(result), 200
                        else:
                            status_code = 400 if 'error' in result else 500
                            return jsonify(result), status_code
                else:
                    logger.warning(f"Source document with nickname '{nickname}' not found for agent '{agent_id}'")
                    # Fall back to RAG query even if source not found (namespace required)
                    if not namespace:
                        return jsonify({
                            'success': False,
                            'answer': None,
                            'metadata': None,
                            'error': 'namespace parameter is required for RAG queries'
                        }), 400
                    
                    result = perform_rag_query(userid, namespace, query, nickname, source_type)
                    
                    if result['success']:
                        return jsonify(result), 200
                    else:
                        status_code = 400 if 'error' in result else 500
                        return jsonify(result), status_code
            except Exception as e:
                logger.error(f"Error updating source document or checking type: {str(e)}", exc_info=True)
                # Don't fail the request if updating example_questions fails, just log it and proceed with RAG
                if not namespace:
                    return jsonify({
                        'success': False,
                        'answer': None,
                        'metadata': None,
                        'error': 'namespace parameter is required for RAG queries'
                    }), 400
                
                result = perform_rag_query(userid, namespace, query, nickname, source_type)
                
                if result['success']:
                    return jsonify(result), 200
                else:
                    status_code = 400 if 'error' in result else 500
                    return jsonify(result), status_code
        else:
            # No nickname provided, just run RAG query without filtering (namespace required)
            if not namespace:
                return jsonify({
                    'success': False,
                    'answer': None,
                    'metadata': None,
                    'error': 'namespace parameter is required for RAG queries'
                }), 400
            
            result = perform_rag_query(userid, namespace, query, '', source_type)
            
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
        logger.error(f"Error in source-confirmed: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'answer': None,
            'metadata': None,
            'error': f'Failed to process source-confirmed request: {str(e)}'
        }), 500

@app.route('/api/handle-rag-message', methods=['POST'])
@require_solari_key
def handle_rag_message():
    """
    Handle RAG message with optional nickname routing.
    
    Expected request body:
    {
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
        
        # Validate required fields
        userid = data.get('userid')
        if not userid:
            return jsonify({
                'success': False,
                'answer': None,
                'metadata': None,
                'error': 'userid parameter is required'
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
                'error': 'query parameter is required'
            }), 400
        
        agent_id = data.get('agent_id')
        if not agent_id:
            return jsonify({
                'success': False,
                'answer': None,
                'metadata': None,
                'error': 'agent_id parameter is required'
            }), 400
        
        # Optional parameters
        nickname = data.get('nickname', '')
        source_type = data.get('source_type', '')
        
        # If nickname is present, check source type and route accordingly
        if nickname:
            logger.info(f"Nickname provided ({nickname}), checking source type...")
            
            # Look up source document to check type
            try:
                db = firestore.client()
                sources_ref = db.collection('users').document(userid).collection('agents').document(agent_id).collection('sources')
                
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
                        'error': f'Source with nickname "{nickname}" not found'
                    }), 400
                
                # Check if source is a table type
                doc_data = source_doc.to_dict()
                source_type_field = doc_data.get('type', '')
                
                if source_type_field == 'table':
                    logger.info(f"Source is a table, routing to /table/ask/ai_sql endpoint...")
                    # Route to table endpoint
                    return query_table_endpoint_internal(userid, source_id, agent_id, query)
                else:
                    # Not a table, proceed with RAG query
                    logger.info(f"Source is not a table, running RAG query...")
                    result = perform_rag_query(userid, namespace, query, nickname, source_type)
                    
                    if result['success']:
                        return jsonify(result), 200
                    else:
                        return jsonify(result), 400
                        
            except Exception as e:
                logger.error(f"Error looking up source: {str(e)}", exc_info=True)
                return jsonify({
                    'success': False,
                    'answer': None,
                    'metadata': None,
                    'error': f'Failed to look up source: {str(e)}'
                }), 500
        
        # If no nickname, run decide_source function first
        else:
            logger.info("No nickname provided, running decide_source function...")
            source_decision = decide_source(userid, agent_id, namespace, query, source_type)
            
            if not source_decision['success']:
                return jsonify({
                    'success': False,
                    'answer': None,
                    'metadata': None,
                    'error': source_decision.get('error', 'Failed to decide source')
                }), 400
            
            # Extract the selected nickname
            selected_nickname = source_decision.get('nickname')
            if not selected_nickname:
                return jsonify({
                    'success': False,
                    'answer': None,
                    'metadata': None,
                    'error': 'No nickname returned from decide_source'
                }), 400
            
            logger.info(f"Selected nickname from decide_source: {selected_nickname}")
            
            # Return only the chosen nickname (don't call ask-pinecone yet)
            return jsonify({
                'success': True,
                'chosen_nickname': selected_nickname,
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
            'error': f'Invalid JSON in request body: {str(e)}'
        }), 400
    except Exception as e:
        logger.error(f"Error handling RAG message: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'answer': None,
            'metadata': None,
            'error': f'Failed to handle RAG message: {str(e)}'
        }), 500


if __name__ == '__main__':
    # Get port from environment or default to 5000
    port = int(get_env_var('PORT', default='5000'))
    debug = get_env_var('FLASK_DEBUG', default='False').lower() == 'true'
    
    app.run(host='0.0.0.0', port=port, debug=debug)

