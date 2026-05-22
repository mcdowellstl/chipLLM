import sys
import os

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

# Monkeypatch streamlit.session_state to be a standard dictionary
class MockSessionState(dict):
    def __getattr__(self, name):
        return self.get(name)
    def __setattr__(self, name, value):
        self[name] = value

st.session_state = MockSessionState()

# Populate with mock tickets
st.session_state.active_tickets = [
    {
        "id": "RC001024",
        "summary": "Kiosk 3 Jammed",
        "status": "Assigned to Field Tech",
        "category": "hardware",
        "subcategory": "kiosk",
        "comments": [{"timestamp": "2026-05-21 10:15:00", "author": "System", "text": "Heartbeat alert"}],
        "created_at": "2026-05-21 10:15:00"
    },
    {
        "id": "RC001025",
        "summary": "Closed ticket yesterday",
        "status": "Closed",
        "category": "hardware",
        "subcategory": "kvs",
        "comments": [],
        "created_at": "2026-05-21 13:45:00"
    },
    {
        "id": "RC001026",
        "summary": "Closed ticket long ago",
        "status": "Closed",
        "category": "hardware",
        "subcategory": "printer",
        "comments": [],
        "created_at": "2026-04-01 15:10:00"
    }
]

from llm_client import get_active_tickets

def test_filtering():
    print("Testing get_active_tickets(status_filter='open'):")
    res_open = get_active_tickets(status_filter="open")
    print(f"Open Tickets Count: {len(res_open['tickets'])}")
    print(f"Open Tickets: {[t['id'] for t in res_open['tickets']]}")
    print(f"Closed in last 30 days: {res_open['closed_last_30_days']}")
    
    # Assertions
    assert len(res_open['tickets']) == 1
    assert res_open['tickets'][0]['id'] == "RC001024"
    assert res_open['closed_last_30_days'] == 1  # Only RC001025 was closed within 30 days (RC001026 is from April)
    
    print("\nTesting get_active_tickets(status_filter='closed'):")
    res_closed = get_active_tickets(status_filter="closed")
    print(f"Closed Tickets Count: {len(res_closed['tickets'])}")
    print(f"Closed Tickets: {[t['id'] for t in res_closed['tickets']]}")
    
    assert len(res_closed['tickets']) == 2
    assert set(t['id'] for t in res_closed['tickets']) == {"RC001025", "RC001026"}
    
    print("\nTesting get_active_tickets(status_filter='all'):")
    res_all = get_active_tickets(status_filter="all")
    print(f"All Tickets Count: {len(res_all['tickets'])}")
    print(f"All Tickets: {[t['id'] for t in res_all['tickets']]}")
    
    assert len(res_all['tickets']) == 3
    
    print("\nAll tests passed successfully!")

if __name__ == "__main__":
    test_filtering()
