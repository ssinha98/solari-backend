import logging
import os
import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://api.pipedrive.com/v1"
HEADERS = {"Content-Type": "application/json"}


def _get_pipedrive_api_key(db, team_id: str) -> str | None:
    """Get Pipedrive API key: first from team doc, then from env."""
    if not team_id:
        return os.getenv("PIPEDRIVE_API_KEY")
    try:
        team_snap = db.collection("teams").document(team_id).get()
        if team_snap.exists:
            team_data = team_snap.to_dict() or {}
            key = team_data.get("PIPEDRIVE_API_KEY") or team_data.get("pipedrive_api_key")
            if key:
                return key
    except Exception as e:
        logger.warning(f"Could not fetch team doc for Pipedrive key: {e}")
    return os.getenv("PIPEDRIVE_API_KEY")


def _get_or_cache_linkedin_field_key(db, team_id: str, api_token: str) -> str:
    """
    Get LinkedIn person field key. Use team doc cache if present;
    otherwise fetch from Pipedrive and persist to team doc.
    """
    team_ref = db.collection("teams").document(team_id)
    team_snap = team_ref.get()
    team_data = (team_snap.to_dict() or {}) if team_snap.exists else {}

    linkedin_hash = team_data.get("PIPEDRIVE_LINKEDIN_HASH") or ""

    updates = {}
    if not linkedin_hash:
        linkedin_hash = get_or_create_person_field("LinkedIn URL", api_token, "varchar")
        updates["PIPEDRIVE_LINKEDIN_HASH"] = linkedin_hash
        logger.info(f"Cached PIPEDRIVE_LINKEDIN_HASH on team {team_id}")

    if updates:
        team_ref.update(updates)

    return linkedin_hash


# ── Step 1: Get or create custom person fields ────────────────────────────────

def get_or_create_person_field(name, api_token, field_type="varchar"):
    """Return the 40-char hash key for a person field, creating it if needed."""
    params = {"api_token": api_token}
    resp = requests.get(f"{BASE_URL}/personFields", headers=HEADERS, params=params)
    resp.raise_for_status()
    fields = resp.json().get("data") or []

    for field in fields:
        if field["name"].lower() == name.lower():
            logger.info(f"Found existing person field '{name}': {field['key']}")
            return field["key"]

    # Field doesn't exist — create it
    resp = requests.post(
        f"{BASE_URL}/personFields",
        json={"name": name, "field_type": field_type},
        headers=HEADERS,
        params=params,
    )
    resp.raise_for_status()
    data = resp.json()
    if data["success"]:
        key = data["data"]["key"]
        logger.info(f"Created person field '{name}': {key}")
        return key
    raise Exception(f"Failed to create person field '{name}': {data}")


# ── Step 2: Get or create organization ───────────────────────────────────────

def get_or_create_organization(company_name, api_token):
    """Return the Pipedrive org ID for a company, creating it if needed."""
    params = {"api_token": api_token, "term": company_name, "exact_match": True}
    resp = requests.get(
        f"{BASE_URL}/organizations/search",
        params=params,
        headers=HEADERS,
    )
    resp.raise_for_status()
    items = (resp.json().get("data") or {}).get("items") or []

    if items:
        org_id = items[0]["item"]["id"]
        logger.info(f"Found existing org '{company_name}': id={org_id}")
        return org_id

    # Org doesn't exist — create it
    params = {"api_token": api_token}
    resp = requests.post(
        f"{BASE_URL}/organizations",
        json={"name": company_name},
        headers=HEADERS,
        params=params,
    )
    resp.raise_for_status()
    data = resp.json()
    if data["success"]:
        org_id = data["data"]["id"]
        logger.info(f"Created org '{company_name}': id={org_id}")
        return org_id
    raise Exception(f"Failed to create org '{company_name}': {data}")


# ── Step 3: Create person ─────────────────────────────────────────────────────

def create_person(name, email, linkedin, org_id, linkedin_field_key, api_token):
    """Create a person in Pipedrive and return their ID."""
    params = {"api_token": api_token}
    resp = requests.post(
        f"{BASE_URL}/persons",
        json={
            "name": name,
            "email": [{"value": email, "primary": True}],
            "org_id": org_id,
            linkedin_field_key: linkedin,
        },
        headers=HEADERS,
        params=params,
    )
    resp.raise_for_status()
    data = resp.json()
    if data["success"]:
        person_id = data["data"]["id"]
        logger.info(f"Created person '{name}': id={person_id}")
        return person_id
    raise Exception(f"Failed to create person '{name}': {data}")


# ── Step 4 & 5: Build and write lead ─────────────────────────────────────────

def create_lead(name, person_id, org_id, api_token):
    """Create a lead in Pipedrive and return its ID."""
    params = {"api_token": api_token}
    resp = requests.post(
        f"{BASE_URL}/leads",
        json={
            "title": name,
            "person_id": person_id,
            "organization_id": org_id,
        },
        headers=HEADERS,
        params=params,
    )
    resp.raise_for_status()
    data = resp.json()
    if data["success"]:
        lead_id = data["data"]["id"]
        logger.info(f"Created lead '{name}': id={lead_id}")
        return lead_id
    raise Exception(f"Failed to create lead '{name}': {data}")


# ── Main entry point ──────────────────────────────────────────────────────────

def write_lead(name, company, email, linkedin, api_token, db=None, team_id=None):
    """
    Full flow to write a lead to Pipedrive.

    Args:
        name       : Contact's full name
        company    : Company / organization name
        email      : Contact's email address
        linkedin   : Contact's LinkedIn URL
        api_token  : Pipedrive API token
        db         : Firestore db (optional) — when provided with team_id, uses team doc cache for field hashes
        team_id    : Team ID (optional) — when provided with db, uses team doc cache for field hashes
    """
    if db and team_id:
        linkedin_field_key = _get_or_cache_linkedin_field_key(db, team_id, api_token)
    else:
        linkedin_field_key = get_or_create_person_field("LinkedIn URL", api_token, "varchar")
    org_id = get_or_create_organization(company, api_token)
    person_id = create_person(name, email, linkedin, org_id, linkedin_field_key, api_token)
    lead_id = create_lead(name, person_id, org_id, api_token)
    return lead_id


def execute(node: dict, variables: dict, db, run_ref, run_data: dict) -> dict:
    """
    Pipedrive node executor.

    Writes a lead to Pipedrive. Reads from node config (after interpolation):
    - pipedriveName, pipedriveCompany, pipedriveEmail, pipedriveLinkedin
    """
    node_data = node.get("data", {})
    label = node_data.get("label", "Pipedrive")
    output_variable = node_data.get("outputVariable")

    name = (node_data.get("pipedriveName") or "").strip()
    company = (node_data.get("pipedriveCompany") or "").strip()
    email = (node_data.get("pipedriveEmail") or "").strip()
    linkedin = (node_data.get("pipedriveLinkedin") or "").strip()
    team_id = run_data.get("teamId")
    api_key = _get_pipedrive_api_key(db, team_id)

    def fail(reason: str):
        return {
            "output": f"failed | {reason}",
            "outputVariable": output_variable,
            "branch": None,
            "pause": False,
            "error": reason,
        }

    if not api_key:
        return fail("PIPEDRIVE_API_KEY not set on team doc or in environment")
    if not name:
        return fail("Pipedrive node has no 'pipedriveName' configured")
    if not company:
        return fail("Pipedrive node has no 'pipedriveCompany' configured")
    if not email:
        return fail("Pipedrive node has no 'pipedriveEmail' configured")

    logger.info(f"Pipedrive node '{label}' writing lead: {name} @ {company}")

    try:
        lead_id = write_lead(name, company, email, linkedin, api_key, db=db, team_id=team_id)
        return {
            "output": f"success | lead_id: {lead_id}",
            "outputVariable": output_variable,
            "branch": None,
            "pause": False,
            "error": None,
        }
    except requests.RequestException as e:
        logger.exception(f"Pipedrive node '{label}' API request failed")
        return fail(str(e))
    except Exception as e:
        logger.exception(f"Pipedrive node '{label}' failed")
        return fail(str(e))