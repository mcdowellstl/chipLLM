"""
mocks/servicenow.py
-------------------
Supabase integrated backend storage layer for incidents/cases.
Provides select, insert, and update operations against a live Supabase database
with a graceful in-memory mock fallback.
"""

import os
import logging
import time
import random
from supabase import create_client, Client

logger = logging.getLogger(__name__)

# Initialize environment variables
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

# Initialize client safely
supabase: Client = None
if not SUPABASE_URL or not SUPABASE_KEY:
    logger.warning("SUPABASE_URL or SUPABASE_KEY is missing from environment. Supabase client disabled; using mock fallback.")
else:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Supabase client initialized successfully.")
    except Exception as e:
        logger.exception("Failed to initialize Supabase client: %s", e)
        supabase = None

# Static in-memory mock database for fallback
MOCK_DATABASE = {
    "67067": [
        {
            "id": "RC001024",
            "store_id": "67067",
            "summary": "Kiosk 3 Cash Acceptor Jammed",
            "status": "Assigned to Field Tech",
            "category": "hardware",
            "subcategory": "kiosk",
            "comments": [
                {"timestamp": "2026-05-21 10:15:00", "author": "System", "text": "Incident opened automatically by device heartbeat alert."},
                {"timestamp": "2026-05-21 11:30:00", "author": "Tech Support", "text": "Dispatching local tech John to site. ETA is 2 hours."}
            ],
            "created_at": "2026-05-21 10:15:00"
        },
        {
            "id": "RC001025",
            "store_id": "67067",
            "summary": "KVS Bumpbar buttons unresponsive in kitchen Zone 1",
            "status": "In Progress",
            "category": "hardware",
            "subcategory": "kvs",
            "comments": [
                {"timestamp": "2026-05-21 13:45:00", "author": "Manager Jim", "text": "Kitchen staff reports keys 3 and 4 are not registering when pressed."},
                {"timestamp": "2026-05-21 14:00:00", "author": "Tech Support", "text": "Rebooted KVS device remotely. Issue persists."}
            ],
            "created_at": "2026-05-21 13:45:00"
        },
        {
            "id": "RC001026",
            "store_id": "67067",
            "summary": "POS 2 receipt printer paper jam sensor failure",
            "status": "New",
            "category": "hardware",
            "subcategory": "printer",
            "comments": [
                {"timestamp": "2026-05-21 15:10:00", "author": "System", "text": "Jam detected in sensor corridor. User cleared jam but sensor remains flagged red."}
            ],
            "created_at": "2026-05-21 15:10:00"
        }
    ],
    "67068": [
        {
            "id": "RC003840",
            "store_id": "67068",
            "summary": "AC unit in dining area blowing warm air",
            "status": "Open",
            "category": "hardware",
            "subcategory": "bos",
            "comments": [],
            "created_at": "2026-05-21 16:00:00"
        }
    ]
}

def _normalize_case_for_app(case: dict) -> dict:
    """
    Ensure the case dictionary has all the fields required by both legacy and modern
    parts of the application (e.g. mapping id/number, summary/short_description, status/state).
    """
    if not case:
        return case
    
    # Create a copy so we do not modify the database records in-place
    norm = dict(case)
    
    # Set default values for missing keys to guarantee full backward compatibility
    if "id" not in norm and "number" in norm:
        norm["id"] = norm["number"]
    if "number" not in norm and "id" in norm:
        norm["number"] = norm["id"]
        
    if "summary" not in norm and "short_description" in norm:
        norm["summary"] = norm["short_description"]
    if "short_description" not in norm and "summary" in norm:
        norm["short_description"] = norm["summary"]
        
    if "status" not in norm and "state" in norm:
        norm["status"] = norm["state"]
    if "state" not in norm and "status" in norm:
        norm["state"] = norm["status"]
        
    if "comments" not in norm:
        norm["comments"] = []
        
    if "priority" not in norm:
        norm["priority"] = "3 - Moderate"
        
    if "sys_created_on" not in norm:
        if "created_at" in norm:
            norm["sys_created_on"] = norm["created_at"]
        else:
            norm["sys_created_on"] = "2026-05-21 12:00:00"
            
    if "category" not in norm:
        norm["category"] = None
    if "subcategory" not in norm:
        norm["subcategory"] = None
    if "sub_category" not in norm:
        norm["sub_category"] = norm["subcategory"]
    if "subcategory" not in norm and "sub_category" in norm:
        norm["subcategory"] = norm["sub_category"]
            
    return norm

def _log_db_exception(func_name: str, e: Exception) -> None:
    """Helper to log detailed exception attributes for Supabase/Postgrest errors."""
    logger.error("DB Exception in %s: type=%s, str=%s", func_name, type(e).__name__, str(e))
    for attr in ["message", "code", "details", "hint", "status"]:
        if hasattr(e, attr):
            logger.error("  Attribute %s: %s", attr, getattr(e, attr))
    if hasattr(e, "__dict__") and e.__dict__:
        try:
            logger.error("  Exception dict: %s", e.__dict__)
        except Exception:
            pass
    logger.exception("Full traceback for database operation failure in %s:", func_name)

def get_active_cases(store_id: str) -> list[dict]:
    """
    Get active incident cases for a given store_id.
    """
    logger.info("=== DB QUERY START: get_active_cases ===")
    logger.info("store_id: %s, Supabase client enabled: %s", store_id, supabase is not None)
    if supabase:
        try:
            logger.info("Executing Supabase select on cases table for store_id: %s", store_id)
            response = supabase.table("cases").select("*").eq("store_id", store_id).order("created_at", desc=False).execute()
            cases = response.data or []
            logger.info("Supabase query successful. Returned %d cases.", len(cases))
            for i, c in enumerate(cases):
                logger.info("  Supabase case %d: ID=%s, Summary=%s, Status=%s, CreatedAt=%s",
                            i, c.get("id"), c.get("summary"), c.get("status"), c.get("created_at"))
            return [_normalize_case_for_app(c) for c in cases]
        except Exception as e:
            _log_db_exception("get_active_cases", e)
            logger.warning("Supabase select failed. Falling back to mock database.")
            
    # Mock fallback
    logger.info("Using mock fallback database for store_id: %s", store_id)
    cases = MOCK_DATABASE.get(store_id, [])
    logger.info("Mock database returned %d cases.", len(cases))
    for i, c in enumerate(cases):
        logger.info("  Mock case %d: ID=%s, Summary=%s, Status=%s, CreatedAt=%s",
                    i, c.get("id"), c.get("summary"), c.get("status"), c.get("created_at"))
    return [_normalize_case_for_app(c) for c in cases]

def get_active_count(store_id: str) -> int:
    """
    Get the count of active cases for a given store_id.
    """
    logger.info("=== DB QUERY START: get_active_count ===")
    logger.info("store_id: %s, Supabase client enabled: %s", store_id, supabase is not None)
    if supabase:
        try:
            response = supabase.table("cases").select("*", count="exact").eq("store_id", store_id).execute()
            if response.count is not None:
                logger.info("Supabase count exact returned: %d", response.count)
                return response.count
            cnt = len(response.data or [])
            logger.info("Supabase fallback list count returned: %d", cnt)
            return cnt
        except Exception as e:
            _log_db_exception("get_active_count", e)
            logger.warning("Supabase count failed. Falling back to mock database count.")
            
    cnt = len(MOCK_DATABASE.get(store_id, []))
    logger.info("Mock count returned: %d", cnt)
    return cnt

def create_case(case_data: dict) -> dict:
    """
    Create a new case in Supabase or local mock storage.
    """
    logger.info("=== DB WRITE START: create_case ===")
    logger.info("Input case_data: %s", case_data)
    # Build standard dictionary for SQL cases table
    db_data = {
        "store_id": case_data.get("store_id"),
        "summary": case_data.get("summary") or case_data.get("short_description") or "New Support Case",
        "status": case_data.get("status") or case_data.get("state") or "New",
        "comments": case_data.get("comments") or [],
        "category": case_data.get("category"),
        "subcategory": case_data.get("subcategory") or case_data.get("sub_category")
    }
    
    if "created_at" in case_data:
        db_data["created_at"] = case_data["created_at"]
        
    logger.info("Normalized DB case data to write: %s", db_data)
        
    if supabase:
        try:
            # If an explicit ID is passed (e.g. mock seeding), pass it to Supabase;
            # otherwise let sequence-based trigger generate it.
            if "id" in case_data:
                db_data["id"] = case_data["id"]
            logger.info("Inserting record into Supabase cases table: %s", db_data)
            response = supabase.table("cases").insert(db_data).execute()
            if response.data:
                res = response.data[0]
                logger.info("Supabase insert successful. Created case in Supabase: %s", res)
                return _normalize_case_for_app(res)
            else:
                logger.warning("Supabase insert response returned empty data.")
        except Exception as e:
            _log_db_exception("create_case", e)
            logger.warning("Supabase create_case failed. Falling back to mock database.")
            
    # Mock fallback
    logger.info("Using mock fallback for case creation.")
    if "id" not in case_data:
        # Mock formatting 'RC######' starting at 1027
        mock_id = f"RC00{random.randint(1027, 9999)}"
        db_data["id"] = mock_id
    else:
        db_data["id"] = case_data["id"]
        
    store_id = db_data["store_id"]
    if store_id not in MOCK_DATABASE:
        MOCK_DATABASE[store_id] = []
        
    MOCK_DATABASE[store_id].append(db_data)
    logger.info("Mock insert successful. Created case: %s", db_data)
    return _normalize_case_for_app(db_data)

def update_case(case_id: str, updates: dict) -> dict:
    """
    Update an existing case in Supabase or local mock storage by ID.
    """
    logger.info("=== DB WRITE START: update_case ===")
    logger.info("case_id: %s, updates: %s", case_id, updates)
    db_updates = {}
    if "summary" in updates:
        db_updates["summary"] = updates["summary"]
    elif "short_description" in updates:
        db_updates["summary"] = updates["short_description"]
        
    if "status" in updates:
        db_updates["status"] = updates["status"]
    elif "state" in updates:
        db_updates["status"] = updates["state"]
        
    if "comments" in updates:
        db_updates["comments"] = updates["comments"]
        
    if "category" in updates:
        db_updates["category"] = updates["category"]
        
    if "subcategory" in updates:
        db_updates["subcategory"] = updates["subcategory"]
    elif "sub_category" in updates:
        db_updates["subcategory"] = updates["sub_category"]
        
    logger.info("Normalized DB updates: %s", db_updates)
        
    if supabase:
        try:
            logger.info("Executing Supabase update on cases table for case_id: %s with updates: %s", case_id, db_updates)
            response = supabase.table("cases").update(db_updates).eq("id", case_id).execute()
            if response.data:
                res = response.data[0]
                logger.info("Supabase update successful. Updated case in Supabase: %s", res)
                return _normalize_case_for_app(res)
            else:
                logger.warning("Supabase update response returned empty data.")
        except Exception as e:
            _log_db_exception("update_case", e)
            logger.warning("Supabase update_case failed. Falling back to mock database.")
            
    # Mock fallback
    logger.info("Using mock fallback for case update.")
    for store_id, cases in MOCK_DATABASE.items():
        for case in cases:
            if case.get("id") == case_id:
                case.update(db_updates)
                logger.info("Mock update successful. Updated case: %s", case)
                return _normalize_case_for_app(case)
                
    logger.warning("Case %s not found in mock database.", case_id)
    return {}

def get_store_tickets(store_id: str) -> list[dict]:
    """
    Get active incident tickets for a given store_id.
    This is retained for legacy compatibility and redirects to get_active_cases.
    """
    return get_active_cases(store_id)

def get_store_mims(store_id: str) -> list[dict]:
    """
    Get major incidents / outages (MIMs) for a given store_id.
    """
    # Provide specific test data for store 67067
    if store_id == "67067":
        return [
            {
                "number": "MIM0008472",
                "short_description": "Regional ISP Fiber Cut - Offline Credit Card Processing",
                "priority": "1 - Critical",
                "description": "A major outage affecting all credit card terminals. Backup cellular routing is active but slow. Please use offline processing mode if transactions fail.",
                "sys_created_on": "2026-05-21 13:05:00"
            }
        ]
    return []
