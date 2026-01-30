import os
import time
from datetime import datetime, timedelta, timezone

import requests
from firebase_admin import firestore

# ======================
# ENV CONFIG
# ======================

WORKER_ID = "upload-worker-1"

WEB_BASE_URL = "https://api.usesolari.ai"
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
    pinecone_slack_upload_internal, 
    _build_jira_ticket_text,
    _merge_jira_tickets,
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

def process_slack_source(db, job_ref, job, source):
    team_id = job["team_id"]
    agent_id = job["agent_id"]
    uid = job["created_by_user_id"]

    channel_id = source.get("id")  # your upload_jobs uses "id" for the channel id
    channel_name = source.get("channel_name") or source.get("title")  # either is fine
    nickname = source.get("nickname") or ""

    source_key = source.get("source_key") or f"slack:{channel_id}"

    # Mark source as processing (uses your existing helper)
    update_source(job_ref, source_key, {
        "status": "processing",
        "stage": "sync_and_embed",
        "error": None,
        "updated_at": utcnow(),
    })

    namespace = _get_team_pinecone_namespace(db, team_id)

    # Reuse your existing Slack ingestion+embedding logic
    result = pinecone_slack_upload_internal(
        db=db,
        uid=uid,
        agent_id=agent_id,
        channel_id=channel_id,
        channel_name=channel_name,
        namespace=namespace,
        nickname=nickname,
        chunk_n=20,
        overlap_n=5,
        team_id=None,              # derive internally / ok to pass None
        source_id=channel_id,
    )

    if not result.get("ok"):
        update_source(job_ref, source_key, {
            "status": "error",
            "stage": "error",
            "error": result.get("error") or "slack_ingest_failed",
            "updated_at": utcnow(),
        })
        return

    update_source(job_ref, source_key, {
        "status": "done",
        "stage": "done",
        "error": None,
        "updated_at": utcnow(),
    })

# ======================
# FIRESTORE JOB HELPERS
# ======================

# def claim_next_job(db):
#     query = (
#         db.collection_group("upload_jobs")
#           .where("status", "==", "queued")
#           .order_by("created_at")
#           .limit(1)
#     )

#     docs = list(query.stream())
#     if not docs:
#         return None

#     job_ref = docs[0].reference

#     @firestore.transactional
#     def txn_claim(txn):
#         snap = job_ref.get(transaction=txn)
#         data = snap.to_dict() or {}

#         if data.get("status") != "queued":
#             return False

#         locked_until = data.get("locked_until")
#         if locked_until and locked_until > utcnow():
#             return False

#         txn.update(job_ref, {
#             "status": "processing",
#             "locked_by": WORKER_ID,
#             "locked_until": utcnow() + timedelta(seconds=LEASE_SECONDS),
#             "message": "Processing",
#             "progress": 0,
#             "updated_at": firestore.SERVER_TIMESTAMP,
#         })
#         return True

#     txn = db.transaction()
#     ok = txn_claim(txn)
#     return job_ref if ok else None

def claim_next_job(db):
    now = utcnow()

    # 1) Prefer queued
    queued_q = (
        db.collection_group("upload_jobs")
          .where("status", "==", "queued")
          .order_by("created_at")
          .limit(1)
    )
    docs = list(queued_q.stream())

    # 2) Otherwise reclaim expired leases
    if not docs:
        expired_q = (
            db.collection_group("upload_jobs")
              .where("status", "==", "processing")
              .where("locked_until", "<=", now)
              .order_by("locked_until")
              .limit(1)
        )
        docs = list(expired_q.stream())

    if not docs:
        return None

    job_ref = docs[0].reference

    @firestore.transactional
    def txn_claim(txn):
        snap = job_ref.get(transaction=txn)
        data = snap.to_dict() or {}

        status = data.get("status")
        if status not in ("queued", "processing"):
            return False

        locked_until = data.get("locked_until")
        if locked_until and locked_until > now:
            return False

        txn.update(job_ref, {
            "status": "processing",
            "locked_by": WORKER_ID,
            "locked_until": now + timedelta(seconds=LEASE_SECONDS),
            "message": "Processing",
            "updated_at": now,
        })
        return True

    return job_ref if txn_claim(db.transaction()) else None

def renew_lease(job_ref):
    job_ref.update({
        "locked_until": utcnow() + timedelta(seconds=LEASE_SECONDS),
        "updated_at": utcnow(),
    })

def update_job(job_ref, **fields):
    fields["updated_at"] = utcnow()
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
        "updated_at": utcnow(),
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
            "updatedAt": utcnow(),
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
            "createdAt": utcnow(),
            "updatedAt": utcnow(),
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
        "updatedAt": utcnow(),
    }

    if agent_doc:
        agent_sources_ref.document(agent_doc.id).update(payload)
        agent_doc_id = agent_doc.id
    else:
        payload["createdAt"] = utcnow()
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
# JIRA INGESTION
# ======================
def process_jira_source(db, job_ref, job, source):
    team_id = job["team_id"]
    agent_id = job["agent_id"]

    now = utcnow()

    # ---- mark source processing ----
    update_source(job_ref, source["source_key"], {
        "status": "processing",
        "stage": "upsert_firestore",
        "error": None,
        "updated_at": now,
    })

    # ---- get ticket payload ----
    ticket = source.get("ticket") or {}
    ticket_id = str(source.get("id") or ticket.get("id") or "")
    if not ticket_id:
        raise RuntimeError("jira_missing_ticket_id")

    # ---- Upsert agent-level Jira source (single doc with tickets array, as you do today) ----
    agent_sources_ref = (
        db.collection("teams").document(team_id)
          .collection("agents").document(agent_id)
          .collection("sources")
    )

    agent_jira_docs = agent_sources_ref.where("type", "==", "jira").limit(1).stream()
    agent_jira_doc = next(agent_jira_docs, None)

    if agent_jira_doc:
        existing_agent_tickets = (agent_jira_doc.to_dict() or {}).get("tickets", [])
        merged = _merge_jira_tickets(existing_agent_tickets, [ticket])
        agent_sources_ref.document(agent_jira_doc.id).update({
            "nickname": "jira",
            "tickets": merged,
            "updated_at": firestore.SERVER_TIMESTAMP,
        })
    else:
        agent_sources_ref.add({
            "type": "jira",
            "nickname": "jira",
            "agent_id": agent_id,
            "tickets": [ticket],
            "created_at": firestore.SERVER_TIMESTAMP,
            "updated_at": firestore.SERVER_TIMESTAMP,
        })

    # ---- Upsert team-level Jira source (single doc with tickets array) ----
    team_sources_ref = db.collection("teams").document(team_id).collection("sources")
    team_jira_docs = team_sources_ref.where("type", "==", "jira").limit(1).stream()
    team_jira_doc = next(team_jira_docs, None)

    if team_jira_doc:
        existing_team_tickets = (team_jira_doc.to_dict() or {}).get("tickets", [])
        merged = _merge_jira_tickets(existing_team_tickets, [ticket])
        team_sources_ref.document(team_jira_doc.id).update({
            "nickname": "jira",
            "tickets": merged,
            "updated_at": firestore.SERVER_TIMESTAMP,
        })
    else:
        team_sources_ref.add({
            "type": "jira",
            "nickname": "jira",
            "tickets": [ticket],
            "created_at": firestore.SERVER_TIMESTAMP,
            "updated_at": firestore.SERVER_TIMESTAMP,
        })

    update_source(job_ref, source["source_key"], {
        "stage": "embed",
        "updated_at": now,
    })

    # ---- Embed + Pinecone ----
    namespace = _get_team_pinecone_namespace(db, team_id)

    text = _build_jira_ticket_text(ticket) or f"Jira Ticket ID {ticket_id}"
    openai_client = get_openai_client()
    emb = generate_embeddings([text], openai_client)[0]

    raw_id = f"jira_{team_id}_{ticket_id}"
    vector_id = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in raw_id)

    vectors = [{
        "id": vector_id,
        "values": emb,
        "metadata": {
            "nickname": "jira",
            "source": "jira",
            "jira_id": ticket_id,
            "text_preview": text[:500],
        },
    }]

    upload_vectors_to_pinecone(vectors, namespace, index_name="production", batch_size=100)

    # ---- done ----
    update_source(job_ref, source["source_key"], {
        "status": "done",
        "stage": "done",
        "checkpoint": {"embedded": True},
        "updated_at": utcnow(),
    })
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
            
        if source["type"] == "slack":
            process_slack_source(db, job_ref, job, source)
            completed += 1
        
        if source["type"] == "jira":
            process_jira_source(db, job_ref, job, source)
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