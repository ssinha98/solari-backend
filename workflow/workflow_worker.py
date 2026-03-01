
import time
import logging
import socket
from firebase_config import db, firestore, auth, storage
from workflow.graph import execute_workflow
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

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    logger.info(f"Workflow worker starting — worker ID: {WORKER_ID}")
    logger.info("Firebase connected, polling for queued runs...")

    while True:
        poll_once(db)
        time.sleep(POLL_INTERVAL_SECONDS)

if __name__ == "__main__":
    main()