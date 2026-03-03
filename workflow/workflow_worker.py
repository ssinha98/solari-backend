import os
import time
import logging
import socket
from firebase_config import db, firestore, auth, storage
from datetime import datetime, timezone
from workflow.graph import execute_workflow
from workflow.deep_research.perplexity import PerplexityAsyncClient
# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────

POLL_INTERVAL_SECONDS = 3
WORKER_ID = socket.gethostname()  # unique per Render instance

# ─── Lock ─────────────────────────────────────────────────────────────────────

def acquire_lock(db, run_ref) -> bool:
    """
    Atomically claim a run using a Firestore transaction.
    Returns True if lock was acquired, False if another worker got there first.
    """
    transaction = db.transaction()

    @firestore.transactional
    def _try_acquire(transaction, run_ref):
        snap = run_ref.get(transaction=transaction)
        if not snap.exists:
            return False

        data = snap.to_dict()

        # only claim if still queued and not already locked
        if data.get("status") != "queued" or data.get("lockedBy"):
            return False

        transaction.update(run_ref, {
            "lockedBy": WORKER_ID,
            "lockedAt": firestore.SERVER_TIMESTAMP,
            "status": "running",
        })
        return True

    try:
        return _try_acquire(transaction, run_ref)
    except Exception as e:
        logger.warning(f"Failed to acquire lock: {e}")
        return False

# ─── Poll ─────────────────────────────────────────────────────────────────────

def poll_once(db):
    """Find one queued run, claim it, and execute it."""
    try:
        runs = (
            db.collection_group("runs")
              .where("status", "==", "queued")
              .order_by("startedAt")
              .limit(1)
              .stream()
        )

        for run in runs:
            run_ref = run.reference
            run_data = run.to_dict()
            run_id = run.id

            logger.info(f"Found queued run: {run_id} for agent: {run_data.get('agentId')}")

            # try to claim it
            acquired = acquire_lock(db, run_ref)
            if not acquired:
                logger.info(f"Run {run_id} already claimed by another worker, skipping")
                return

            logger.info(f"Claimed run {run_id}, starting execution")

            try:
                execute_workflow(db, run_ref, run_data)
                logger.info(f"Run {run_id} completed successfully")

            except Exception as e:
                logger.error(f"Run {run_id} failed: {e}", exc_info=True)
                run_ref.update({
                    "status": "failed",
                    "failureReason": str(e),
                    "failedAt": firestore.SERVER_TIMESTAMP,
                    "lockedBy": None,
                })

    except Exception as e:
        logger.error(f"Poll error: {e}", exc_info=True)


def poll_deep_research_once(db):
    """
    Find runs paused with deepResearchJob (Perplexity), check job status.
    When COMPLETED: write node output, update variables, clear deepResearchJob, re-queue run.
    When FAILED: mark run failed, clear deepResearchJob.
    """
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        return

    try:
        runs = (
            db.collection_group("runs")
            .where("status", "==", "paused_deep_research")
            .limit(20)
            .stream()
        )
        for run in runs:
            run_ref = run.reference
            run_data = run.to_dict()
            job = run_data.get("deepResearchJob")
            if not job or job.get("provider") != "perplexity deep research":
                continue

            run_id = run.id
            thread_id = job.get("threadId")
            node_id = job.get("nodeId")
            output_variable = job.get("outputVariable")
            if not thread_id or not node_id:
                logger.warning(f"Run {run_id} deepResearchJob missing threadId or nodeId, skipping")
                continue

            client = PerplexityAsyncClient(api_key=api_key)
            try:
                status_data = client.check_job_status(thread_id)
            except Exception as e:
                logger.warning(f"Run {run_id} check_job_status failed: {e}")
                continue

            status = status_data.get("status")

            if status == "COMPLETED":
                response = status_data.get("response", {})
                content = (response.get("choices") or [{}])[0].get("message", {}) or {}
                content = content.get("content", "") or ""

                raw_citations = response.get("citations", [])
                citations = []
                for c in raw_citations:
                    if isinstance(c, dict):
                        citations.append({"title": c.get("title"), "url": c.get("url")})
                    elif isinstance(c, str):
                        citations.append({"title": None, "url": c})

                citation_lines = [
                    f"[{i + 1}] {c.get('url') or ''}" for i, c in enumerate(citations)
                ]
                full_response = content
                if citation_lines:
                    full_response = content.rstrip() + "\n\n" + "\n".join(citation_lines)

                logger.info(f"Run {run_id} deep research full output:\n{full_response}")

                agent_ref = run_ref.parent.parent
                version_id = run_data.get("versionId")
                version_snap = agent_ref.collection("versions").document(version_id).get()
                if not version_snap.exists:
                    logger.error(f"Run {run_id} version {version_id} not found")
                    continue
                version = version_snap.to_dict()
                nodes = version.get("nodes", [])
                node = next((n for n in nodes if n.get("id") == node_id), None)
                if not node:
                    logger.error(f"Run {run_id} node {node_id} not found in version")
                    continue

                # Update the existing node execution (written when node paused) with final output
                run_snap = run_ref.get()
                run_data_fresh = run_snap.to_dict() or {}
                executions = list(run_data_fresh.get("nodeExecutions") or [])
                for i, ex in enumerate(executions):
                    if ex.get("nodeId") == node_id:
                        executions[i] = {
                            **ex,
                            "output": full_response,
                            "status": "completed",
                            "completedAt": datetime.now(timezone.utc),
                        }
                        break
                updates = {
                    "nodeExecutions": executions,
                    "deepResearchJob": firestore.DELETE_FIELD,
                    "status": "queued",
                    "resumeFromNodeId": node_id,
                    "updatedAt": firestore.SERVER_TIMESTAMP,
                    "citations": citations,
                }
                if output_variable:
                    updates[f"variables.{output_variable}"] = full_response
                    updates[f"outputVariables.{output_variable}"] = full_response

                run_ref.update(updates)
                logger.info(f"Run {run_id} deep research completed, re-queued from node {node_id}")

            elif status == "FAILED":
                error_msg = status_data.get("error_message", "Unknown error")
                run_ref.update({
                    "status": "failed",
                    "failureReason": error_msg,
                    "failedAt": firestore.SERVER_TIMESTAMP,
                    "deepResearchJob": firestore.DELETE_FIELD,
                    "updatedAt": firestore.SERVER_TIMESTAMP,
                })
                logger.info(f"Run {run_id} deep research failed: {error_msg}")
            # else: still processing, next poll will check again

    except Exception as e:
        logger.error(f"Deep research poll error: {e}", exc_info=True)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    logger.info(f"Workflow worker starting — worker ID: {WORKER_ID}")
    logger.info("Firebase connected, polling for queued runs...")

    while True:
        poll_deep_research_once(db)
        poll_once(db)
        time.sleep(POLL_INTERVAL_SECONDS)

if __name__ == "__main__":
    main()