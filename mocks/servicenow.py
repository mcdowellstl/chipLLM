"""
mocks/servicenow.py
-------------------
Mock module for ServiceNow integration. Provides static mock data for active
incidents and major incidents (MIMs) filtered by store string.
"""

def get_store_tickets(store_id: str) -> list[dict]:
    """
    Get active incident tickets for a given store_id.
    """
    if store_id == "67067":
        return [
            {
                "id": "INC-001024",
                "number": "INC-001024",
                "summary": "Kiosk 3 Cash Acceptor Jammed",
                "short_description": "Kiosk 3 Cash Acceptor Jammed",
                "status": "Assigned to Field Tech",
                "state": "Assigned to Field Tech",
                "priority": "2 - High",
                "sys_created_on": "2026-05-21 10:15:00",
                "comments": [
                    {"timestamp": "2026-05-21 10:15:00", "author": "System", "text": "Incident opened automatically by device heartbeat alert."},
                    {"timestamp": "2026-05-21 11:30:00", "author": "Tech Support", "text": "Dispatching local tech John to site. ETA is 2 hours."}
                ]
            },
            {
                "id": "INC-001025",
                "number": "INC-001025",
                "summary": "KVS Bumpbar buttons unresponsive in kitchen Zone 1",
                "short_description": "KVS Bumpbar buttons unresponsive in kitchen Zone 1",
                "status": "In Progress",
                "state": "In Progress",
                "priority": "3 - Moderate",
                "sys_created_on": "2026-05-21 13:45:00",
                "comments": [
                    {"timestamp": "2026-05-21 13:45:00", "author": "Manager Jim", "text": "Kitchen staff reports keys 3 and 4 are not registering when pressed."},
                    {"timestamp": "2026-05-21 14:00:00", "author": "Tech Support", "text": "Rebooted KVS device remotely. Issue persists."}
                ]
            },
            {
                "id": "INC-001026",
                "number": "INC-001026",
                "summary": "POS 2 receipt printer paper jam sensor failure",
                "short_description": "POS 2 receipt printer paper jam sensor failure",
                "status": "New",
                "state": "New",
                "priority": "4 - Low",
                "sys_created_on": "2026-05-21 15:10:00",
                "comments": [
                    {"timestamp": "2026-05-21 15:10:00", "author": "System", "text": "Jam detected in sensor corridor. User cleared jam but sensor remains flagged red."}
                ]
            }
        ]
    elif store_id == "67068":
        return [
            {
                "number": "INC0038401",
                "short_description": "AC unit in dining area blowing warm air",
                "priority": "3 - Moderate",
                "state": "Open",
                "sys_created_on": "2026-05-21 16:00:00"
            }
        ]
    return []

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
