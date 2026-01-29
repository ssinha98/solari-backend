import os
import time
from datetime import datetime, timedelta, timezone

import requests
from google.cloud import firestore

# ======================
# ENV CONFIG
# ======================

WORKER_ID = "upload-worker-1"

WEB_BASE_URL = "api.usesolari.ai"
SOLARI_INTERNAL_KEY = os.environ["SOLARI_INTERNAL_KEY"]

POLL_SECONDS = float(os.environ.get("JOB_POLL_SECONDS", "2"))
LEASE_SECONDS = int(os.environ.get("JOB_LEASE_SECONDS", "120"))
LEASE_RENEW_SECONDS = int(os.environ.get("JOB_LEASE_RENEW_SECONDS", "30"))

DEFAULT_CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1000"))
DEFAULT_CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "200"))

# PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "production")

# ======================
# HELPERS YOU ALREADY HAVE
# (these must be importable)
# ======================

from app import (
    chunk_text,
    generate_embeddings,
    upload_vectors_to_pinecone,
    upload_string_to_firebase_storage,
    confluence_storage_html_to_rag_text,
    get_openai_client,
    _get_team_pinecone_namespace,
)

# ======================
# TIME HELPERS
# ======================

def utcnow():
    return datetime.now(timezone.utc)

# ======================
# CONFLUENCE FETCH (WORKER SAFE)
# ======================

def fetch_confluence_storage_html_for_worker(user_id: str, page_id: str) -> str:
    url = f"{WEB_BASE_URL}/api/confluence/get-page"
    headers = {
        "Accept": "application/json",
        "x-solari-key": SOLARI_INTERNAL_KEY,
    }
    params = {
        "user_id": user_id,
        "page_id": page_id,
        "body_format": "storage",
    }

    resp = requests.get(url, headers=headers, params=params, timeout=45)
    if resp.status_code != 200:
        raise Exception(f"Confluence fetch failed: {resp.text}")

    page = (resp.json() or {}).get("page") or {}
    body = (page.get("body") or {}).get("storage") or {}
    return body.get("value") or ""

# ======================
# FIRESTORE JOB HELPERS
# ======================

def claim_next_job(db):
    query = (
        db.collection_group("upload_jobs")
          .where("status", "==", "queued")
          .order_by("created_at")
          .limit(1)
    )

    docs = list(query.stream())
    if not docs:
        return None

    job_ref = docs[0].reference

    @firestore.transactional
    def txn_claim(txn):
        snap = job_ref.get(transaction=txn)
        data = snap.to_dict() or {}

        if data.get("status") != "queued":
            return False

        locked_until = data.get("locked_until")
        if locked_until and locked_until > utcnow():
            return False

        txn.update(job_ref, {
            "status": "processing",
            "locked_by": WORKER_ID,
            "locked_until": utcnow() + timedelta(seconds=LEASE_SECONDS),
            "message": "Processing",
            "progress": 0,
            "updated_at": firestore.SERVER_TIMESTAMP,
        })
        return True

    txn = db.transaction()
    ok = txn_claim(txn)
    return job_ref if ok else None

def renew_lease(job_ref):
    job_ref.update({
        "locked_until": utcnow() + timedelta(seconds=LEASE_SECONDS),
        "updated_at": firestore.SERVER_TIMESTAMP,
    })

def update_job(job_ref, **fields):
    fields["updated_at"] = firestore.SERVER_TIMESTAMP
    job_ref.update(fields)

def update_source(job_ref, source_key, patch):
    snap = job_ref.get()
    job = snap.to_dict() or {}
    sources = job.get("sources", [])

    for i, s in enumerate(sources):
        if s.get("source_key") == source_key:
            sources[i] = {**s, **patch}
            break

    job_ref.update({
        "sources": sources,
        "updated_at": firestore.SERVER_TIMESTAMP,
    })

# ======================
# CONFLUENCE INGESTION
# ======================

def process_confluence_source(db, job_ref, job, source):
    team_id = job["team_id"]
    agent_id = job["agent_id"]
    user_id = job["created_by_user_id"]

    namespace = _get_team_pinecone_namespace(db, team_id)
    openai_client = get_openai_client()

    agent_sources_ref = (
        db.collection("teams").document(team_id)
          .collection("agents").document(agent_id)
          .collection("sources")
    )
    team_sources_ref = db.collection("teams").document(team_id).collection("sources")

    page_id = source["id"]
    title = source.get("title") or page_id
    tinyui_path = source.get("tinyui_path")
    url = source.get("url")
    excerpt = source.get("excerpt")
    nickname = title
    source_key = source["source_key"]

    update_source(job_ref, source_key, {"status": "processing", "stage": "fetch"})

    storage_html = fetch_confluence_storage_html_for_worker(user_id, page_id)
    processed_text = confluence_storage_html_to_rag_text(storage_html) if storage_html else ""

    # ---- TEAM SOURCE UPSERT ----
    team_match = team_sources_ref.where("id", "==", page_id).limit(1).stream()
    team_doc = next(team_match, None)

    if team_doc:
        team_source_id = team_doc.id
        team_sources_ref.document(team_source_id).update({
            "title": title,
            "id": page_id,
            "tinyui_path": tinyui_path,
            "url": url,
            "type": "confluence",
            "bodyPreview": excerpt,
            "nickname": nickname,
            "agents": firestore.ArrayUnion([agent_id]),
            "updatedAt": firestore.SERVER_TIMESTAMP,
        })
    else:
        ref = team_sources_ref.document()
        team_source_id = ref.id
        ref.set({
            "title": title,
            "id": page_id,
            "tinyui_path": tinyui_path,
            "url": url,
            "type": "confluence",
            "bodyPreview": excerpt,
            "nickname": nickname,
            "agents": [agent_id],
            "createdAt": firestore.SERVER_TIMESTAMP,
            "updatedAt": firestore.SERVER_TIMESTAMP,
        })

    # ---- AGENT SOURCE UPSERT ----
    agent_match = agent_sources_ref.where("id", "==", page_id).limit(1).stream()
    agent_doc = next(agent_match, None)

    payload = {
        "title": title,
        "id": page_id,
        "tinyui_path": tinyui_path,
        "url": url,
        "type": "confluence_page",
        "bodyPreview": excerpt,
        "nickname": nickname,
        "updatedAt": firestore.SERVER_TIMESTAMP,
    }

    if agent_doc:
        agent_sources_ref.document(agent_doc.id).update(payload)
        agent_doc_id = agent_doc.id
    else:
        payload["createdAt"] = firestore.SERVER_TIMESTAMP
        agent_sources_ref.add(payload)
        agent_doc_id = None

    # ---- STORAGE ----
    raw_path = f"teams/{team_id}/sources/{team_source_id}/confluence/pages/{page_id}/raw_storage.html"
    processed_path = f"teams/{team_id}/sources/{team_source_id}/confluence/pages/{page_id}/processed.txt"

    upload_string_to_firebase_storage(raw_path, storage_html, "text/html")
    upload_string_to_firebase_storage(processed_path, processed_text, "text/plain")

    update_source(job_ref, source_key, {"stage": "embed"})

    if not processed_text.strip():
        update_source(job_ref, source_key, {"status": "done", "stage": "done"})
        return

    chunks = chunk_text(processed_text, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP)

    checkpoint = source.get("checkpoint") or {}
    start_i = int(checkpoint.get("chunk_index") or 0)

    for i in range(start_i, len(chunks)):
        emb = generate_embeddings([chunks[i]], openai_client)[0]

        upload_vectors_to_pinecone(
            [{
                "id": f"confluence_{team_source_id}_{page_id}_chunk_{i}",
                "values": emb,
                "metadata": {
                    "source": "confluence",
                    "nickname": nickname,
                    "confluence_page_id": page_id,
                    "source_id": team_source_id,
                    "text_preview": chunks[i][:500],
                }
            }],
            namespace=namespace,
            index_name="production",
            batch_size=1
        )

        update_source(job_ref, source_key, {
            "checkpoint": {
                "chunk_index": i + 1,
                "total_chunks": len(chunks),
            }
        })

    update_source(job_ref, source_key, {"status": "done", "stage": "done"})

# ======================
# JOB PROCESSOR
# ======================

def process_job(db, job_ref):
    job = job_ref.get().to_dict() or {}
    sources = job.get("sources", [])

    completed = 0
    last_renew = time.time()

    for source in sources:
        if time.time() - last_renew > LEASE_RENEW_SECONDS:
            renew_lease(job_ref)
            last_renew = time.time()

        if source["status"] == "done":
            completed += 1
            continue

        if source["type"] == "confluence":
            process_confluence_source(db, job_ref, job, source)
            completed += 1

        update_job(
            job_ref,
            progress=int((completed / len(sources)) * 100),
            message=f"Completed {completed}/{len(sources)} sources",
        )

    update_job(job_ref, status="done", progress=100, message="Completed")

# ======================
# MAIN LOOP
# ======================

def main():
    db = firestore.client()
    print(f"[worker] started as {WORKER_ID}")

    while True:
        job_ref = claim_next_job(db)
        if not job_ref:
            time.sleep(POLL_SECONDS)
            continue

        try:
            process_job(db, job_ref)
        except Exception as e:
            update_job(job_ref, status="error", message=str(e))
        finally:
            update_job(job_ref, locked_by=None, locked_until=None)

if __name__ == "__main__":
    main()