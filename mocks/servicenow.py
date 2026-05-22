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
    # Provide specific test data for store 67067
    if store_id == "67067":
        return [
            {
                "number": "INC0038291",
                "short_description": "KDS Screen unresponsive in Zone 2",
                "priority": "2 - High",
                "state": "In Progress",
                "sys_created_on": "2026-05-21 14:32:10"
            },
            {
                "number": "INC0038295",
                "short_description": "POS 2 receipt printer paper jam sensor failure",
                "priority": "3 - Moderate",
                "state": "New",
                "sys_created_on": "2026-05-21 15:10:45"
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
