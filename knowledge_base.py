"""
knowledge_base.py
-----------------
In-memory RAG knowledge base for chipLLM.
Each entry contains:
  - keywords: list of trigger terms for lightweight fuzzy matching
  - title:    human-readable playbook title
  - content:  full troubleshooting playbook injected into context window
"""

import os
import re
import logging
from bs4 import BeautifulSoup
from google.cloud import storage

_log = logging.getLogger("chipLLM.knowledge_base")

# ---------------------------------------------------------------------------
# Fallback hardcoded playbooks (used if L0_KA is empty)
# ---------------------------------------------------------------------------
FALLBACK_PLAYBOOKS: list[dict] = [
    {
        "id": "POS_PRINTER_OFFLINE",
        "keywords": [
            "printer", "print", "receipt", "offline", "jam", "jammed",
            "paper", "stuck", "no print", "won't print", "can't print",
            "printer error", "receipt printer",
        ],
        "title": "POS Receipt Printer – Offline / Paper Jam Recovery",
        "content": """
## Playbook: POS Receipt Printer – Offline / Paper Jam Recovery
**Asset Types:** Epson TM-T88VI, Star TSP143, Bixolon SRP-350

### Symptoms
- POS terminal shows "Printer Offline" or "Check Printer" alert
- Receipt paper not advancing; audible grinding or clicking
- LED status light solid red or rapid amber flash
- Print jobs queuing but not executing

### Immediate Triage (60 seconds)
1. **Power cycle sequence:** Press and hold the FEED button, then power OFF. Wait 10 seconds. Power ON while holding FEED until LED blinks twice.
2. **Open the paper cover** completely. Remove all paper. Look for torn paper fragments near the thermal head—use tweezers (NOT metal tools near the head).
3. **Inspect the cutter blade:** Rotate manually 1/4 turn clockwise if visibly jammed. Never force.
4. **Reload paper:** Ensure paper feeds from the BOTTOM of the roll (thermal side up). Close the cover until it clicks audibly.
5. **Test print:** On the POS terminal, navigate to Settings → Peripherals → Printer → Test Print.

### Network/USB Connectivity Check
- **USB:** Unplug USB from printer, replug into a different USB port on the POS terminal. Check Device Manager (POS terminal) for yellow exclamation mark.
- **Ethernet (kitchen/expo printers):** Ping the printer IP (found on config label on printer underside). Default IPs: 192.168.1.100–110. If no ping response, hold FEED + POWER to print self-test showing current IP config.
- **Wi-Fi models:** Check signal strength LED (solid green = connected). Re-run Auto Setup from the printer's WPS button if flashing.

### Driver & Port Reset (Windows POS)
1. Open **Devices and Printers** → right-click the printer → Remove Device.
2. In POS software (e.g., Aloha, Toast, Square), go to Back Office → Hardware → Printers → Delete and re-add the printer using COM port or IP.
3. Restart POS Print Spooler: `net stop spooler && net start spooler`

### Escalation Criteria
- Paper jam persists after 3 power cycle attempts → **Hardware swap required**
- Burn marks or smell from thermal head → **Immediately power off, do not swap paper, escalate to depot**
- Self-test print shows garbled characters → Firmware mismatch; contact vendor support

### Metadata Tags
`severity: medium | asset_type: PRINTER | sla: 2h`
""",
    },
    {
        "id": "WAYSTATION_BOOT_LOOP",
        "keywords": [
            "waystation", "way station", "kvs", "kitchen video", "display",
            "boot", "reboot", "loop", "looping", "restart", "bootloop",
            "boot loop", "screen flicker", "black screen", "monitor",
            "kitchen display", "expo display", "kds",
        ],
        "title": "Waystation / KDS – Boot Loop & Display Recovery",
        "content": """
## Playbook: Waystation / KDS – Boot Loop & Display Recovery
**Asset Types:** HME ZOOM KDS, Makeline Display, Par Brink KDS, Oracle MICROS KDS

### Symptoms
- Display unit continuously reboots (POST screen → black → repeat)
- Splash screen freezes mid-boot for >90 seconds
- Screen flickers, displays vertical/horizontal lines, or shows solid color
- Unit powers on but shows "No Signal" on attached monitor
- Order items not appearing despite POS orders being placed

### Step 1 – Hard Reset Sequence
1. Hold the physical POWER button for 15 seconds (full shutdown, not soft reset).
2. Disconnect power from the wall (or PoE switch port). Wait 30 seconds.
3. If the unit has a removable battery backup (PAX models), disconnect it now.
4. Reconnect power. Do NOT press any buttons during initial POST (30 seconds).

### Step 2 – Storage & Firmware Integrity Check
- **SD Card / SSD models:** Remove the storage card (panel on rear). Inspect for physical damage. Reseat firmly. If unit has dual-card, swap to secondary.
- **Firmware boot flag:** Connect a USB keyboard. When the PAR logo appears, press `F11` to enter Boot Options. Select "Recovery Partition" if available.
- **Log retrieval:** Once booted (even partially), SSH into unit at `waystation.local` or `192.168.1.2X`. Run `journalctl -xe --no-pager | tail -200` to capture boot errors.

### Step 3 – Network & POS Sync Validation
1. Confirm KDS is on the same VLAN as POS terminals (typically VLAN 10 on managed switches).
2. Ping the KDS from the POS terminal: `ping 192.168.10.5X`
3. In POS back office, navigate to KDS Configuration → Verify "Station ID" matches the physical unit's sticker.
4. Resync order routing: Back Office → Kitchen Routing → Reset → Apply.

### Step 4 – Monitor / Display Separation
- **Black screen with unit responsive (ping succeeds):** HDMI/DP cable issue. Swap cable first. Try alternate HDMI port. Test with a known-good monitor.
- **Flickering:** Disable hardware acceleration in KDS software config. Update GPU driver via vendor update portal.

### Nuclear Option – Factory Reset
> ⚠️ Only with TL approval. This clears all local config.
1. Boot to recovery via USB recovery drive (request from depot).
2. Select "Factory Reset – Keep Network."
3. Re-provision using MDM (Jamf / Intune) auto-enrollment. Unit should self-configure within 15 minutes on first boot.

### Escalation Criteria
- Boot loop persists after Step 1–2 → escalate to **Tier 2 Hardware** with boot logs
- Physical screen damage (cracks, dead pixels) → **RMA immediately**

### Metadata Tags
`severity: high | asset_type: KDS | sla: 1h`
""",
    },
    {
        "id": "POS_TERMINAL_OFFLINE",
        "keywords": [
            "pos", "terminal", "register", "toast", "aloha", "square", "brink",
            "micros", "crash", "frozen", "hangs", "slow", "not responding",
            "can't log in", "login", "drawer", "cash drawer", "transaction",
            "payment", "card reader", "swipe", "chip reader", "emv",
        ],
        "title": "POS Terminal – Crash, Freeze & Payment Hardware Recovery",
        "content": """
## Playbook: POS Terminal – Crash, Freeze & Payment Hardware Recovery
**Asset Types:** Elo Touch I-Series, Aures SANGO, PAX IM30, Oracle MICROS Workstation

### Symptoms
- POS application crashes to desktop or black screen mid-service
- Terminal frozen on order screen; touch input unresponsive
- Card reader (EMV/NFC) not detecting cards or returning "Read Error"
- Cash drawer not opening after successful transaction
- Login screen loops or throws "License Server Unreachable"

### Immediate Recovery (< 2 minutes)
1. **Soft restart the POS application:**
   - Aloha: `Ctrl+Alt+Delete` → Task Manager → End Task "Aloha.exe" → Relaunch from desktop shortcut
   - Toast: Long-press home button → Close app → Reopen from App Drawer
   - Square: Settings → Software → Restart POS (preserves offline mode queue)
2. **If touch is unresponsive:** Plug in a USB mouse to perform the above steps.
3. **Full reboot:** Hold power 5 seconds → Restart. Do NOT force shutdown during active transactions (risk of DB corruption).

### Payment Hardware – EMV / NFC / MSR
1. **Unplug the payment terminal** (Verifone, Ingenico, PAX) from USB/serial. Wait 10 seconds. Replug.
2. In POS back office: Settings → Payment → Re-initialize Terminal.
3. **Test transaction:** Run a $0.01 test charge on a test card. If declined with error code, note the 3-digit code:
   - `052` → Card reader firmware mismatch. Update via vendor portal.
   - `099` → Processor connectivity issue. Check internet/firewall rules (port 443 to payment gateway IP).
   - `067` → Card physically damaged. Try alternate card.
4. **NFC/Contactless not working:** Disable and re-enable NFC in payment terminal settings. Ensure no metal objects or magnets within 2 inches of the reader.

### Cash Drawer Recovery
1. Check the RJ11/RJ12 cable between POS and drawer. Reseat both ends.
2. In POS software, manually trigger drawer: Settings → Peripherals → Open Cash Drawer.
3. If still not opening, insert the manual key (taped to drawer underside in most installations) and rotate 90° to open.
4. Check drawer solenoid with a multimeter at 24V DC input. No voltage = POS board issue.

### License / Server Connectivity
- Run `nslookup license.posvendor.com` from POS terminal. If it fails → DNS issue on store network.
- Check firewall rules. POS license servers typically require outbound 443 and 8443.
- Call vendor license line with terminal's `Machine ID` (found in Help → About).

### Data Protection Note
> Never power-off a POS terminal forcefully during end-of-day reconciliation. Always use the in-app "Close Day" function first.

### Escalation Criteria
- Repeated crashes (3+ in a shift) after reboot → Pull diagnostic logs from `C:\\Aloha\\Logs` or `/var/log/pos/` and submit to Tier 2
- Payment hardware unresponsive after swap → **PCI incident protocol**, contact IT Security

### Metadata Tags
`severity: critical | asset_type: POS_TERMINAL | sla: 30m`
""",
    },
    {
        "id": "KIOSK_SELF_ORDER",
        "keywords": [
            "kiosk", "self order", "self-order", "self service", "self-service",
            "kiosk screen", "kiosk frozen", "kiosk offline", "customer kiosk",
            "order kiosk", "payment kiosk", "kiosk crash", "kiosk reboot",
            "loyalty", "coupon", "qr code", "kiosk printer",
        ],
        "title": "Self-Order Kiosk – Freeze, Payment & Printer Recovery",
        "content": """
## Playbook: Self-Order Kiosk – Freeze, Payment & Printer Recovery
**Asset Types:** Tillster Kiosk, ACRELEC GK, Elo Self-Order, Pyramid Impulse

### Symptoms
- Kiosk touchscreen frozen on menu or payment screen
- "Out of Order" mode activated unexpectedly
- Customer-facing receipt printer jammed
- Loyalty/QR code scanner not reading
- Kiosk rebooting between orders (thermal throttle or power issue)

### Step 1 – Out-of-Order State Recovery
1. Locate the **manager keypad** or **hidden touchpoint** (typically bottom-right corner, 3-finger tap hold for 3 seconds).
2. Enter manager PIN (default: `1234` or store-specific 6-digit code from your FOH binder).
3. Navigate to: Admin → System → Force Restart Application.
4. If Admin panel is inaccessible, perform hard reboot: Open the rear access panel (key from binder), hold power button 10 seconds.

### Step 2 – Payment Module Recovery
1. The kiosk payment module (Ingenico iSMP4 or Verifone e285) can be hot-swapped in most models.
2. **To hot-swap:** Open the side payment door, pull the payment module straight out (no screws), insert replacement, wait 60 seconds for POS sync.
3. In kiosk admin: Payment → Re-pair Module → follow on-screen pairing wizard.

### Step 3 – Kiosk Receipt Printer
- Kiosk printers are typically Star Micronics variants. Follow the same jam recovery steps as the POS Printer playbook.
- Note: Kiosk printers often have a **low-paper sensor** that triggers Out-of-Order mode. Check the paper level first.
- Paper roll replacement: Use 80mm thermal paper, BPA-free (required for customer-facing units per health code in some states).

### Step 4 – Scanner / QR Code / Loyalty
1. Unplug the scanner USB. Replug.
2. Clean the scanner glass with a dry microfiber cloth.
3. Test with a printed QR code (not a phone screen—glare causes false negatives).
4. In kiosk admin: Peripherals → Scanner → Run Diagnostic.

### Step 5 – Thermal Throttle / Spontaneous Reboots
1. Check the kiosk internal temperature via Admin → Diagnostics → Hardware Monitor. Threshold: >75°C triggers auto-reboot.
2. Ensure air vents (rear/bottom) are not blocked by signage, bags, or debris.
3. If ambient temp in lobby exceeds 85°F, reduce screen brightness to 60% to lower thermal load.

### Escalation Criteria
- Payment module swap does not resolve issue → **PCI incident log required**
- Screen physically cracked → **RMA + remove from service immediately**
- 3+ spontaneous reboots in 1 hour after vent clearing → Suspected PSU failure, escalate to depot

### Metadata Tags
`severity: high | asset_type: KIOSK | sla: 1h`
""",
    },
    {
        "id": "DRIVETHRU_BOARD_DAMAGE",
        "keywords": [
            "drive thru", "drive-thru", "drive through", "drivethrough",
            "menu board", "menuboard", "outdoor board", "outdoor sign",
            "digital sign", "digital signage", "outdoor display",
            "sign", "signage", "outdoor menu",
            "crashed into", "car crashed", "vehicle hit", "hit the sign",
            "hit the board", "damaged sign", "damaged board", "knocked over",
            "knocked down", "physical damage", "broken sign", "broken board",
            "drivethru board", "drive-thru board", "outdoor kiosk",
            "order board", "speaker post", "order speaker", "loop detector",
            "drivethru display", "drivethru screen",
        ],
        "title": "Drive-Thru Menu Board / Outdoor Signage – Physical Damage & Recovery",
        "content": """
## Playbook: Drive-Thru Menu Board / Outdoor Signage – Physical Damage & Recovery
**Asset Types:** Delphi Display Systems, HME Drive-Thru Board, Xenon Digital Menuboard, PAR Drive-Thru Display

### Symptoms
- Vehicle impact or physical collision with drive-thru board, speaker post, or order station
- Board not powering on / blank / partial display after physical damage
- Outdoor display shows image artifacts, cracked panel, or no signal
- Speaker post or order station tilted, uprooted, or structurally compromised
- Loop detector (in-ground car sensor) not registering vehicles

### Immediate Response (Safety First)
1. **Secure the area:** If the post is leaning or unstable, place traffic cones and do NOT attempt to operate drive-thru lane until secured.
2. **Document the damage:** Take photos of the board, post, and surrounding area for the insurance and vendor claim.
3. **Power down:** Locate the outdoor equipment circuit breaker (typically in the back-of-house electrical panel, labeled "DT BOARD" or "MENUBOARD") and switch it OFF to prevent electrical hazard from a damaged unit.

### Power & Display Check (if unit appears structurally intact)
1. **Inspect power cable** at the base of the post for cuts or exposed wiring. Do NOT touch exposed wiring—call an electrician if found.
2. **Power cycle from breaker:** Off for 30 seconds, then back on. Wait 2 minutes for the board controller to boot.
3. **Check signal cable:** HDMI/RS-422 cable connecting the board controller box (usually inside the restaurant) to the outdoor display. Reseat both ends.
4. **Controller box:** Located inside at the drive-thru station or manager's office. Check for solid power LED. If blinking red, the board detected a hardware fault — note the blink code and escalate.

### Software / Connectivity
- **No content showing (blank board, power OK):** Log into the menuboard management software (Xenon Portal / HME Cloud / Delphi Connect) and push a content refresh.
- **Board offline in management portal:** Verify the LAN/cellular connection on the controller box. Check the SIM card or ethernet run from store router to the outdoor cabinet.
- **Wrong content / outdated pricing:** Menuboard CMS is typically managed by the marketing team — escalate a content sync request through your regional contact.

### Escalation Criteria
- Any structural damage to the post → **Field hardware vendor dispatch required** — do not attempt repair
- Exposed electrical wiring → **Call licensed electrician before powering on**
- Board controller shows fault blink code → Escalate to drive-thru vendor support with the blink count
- Damage caused by third party (vehicle) → Ensure an incident report is filed with the store manager and escalate to facilities

### Metadata Tags
`severity: high | asset_type: DRIVETHRU_BOARD | sla: 4h`
""",
    },
]


def _load_playbooks_from_html() -> list[dict]:
    playbooks = []
    dir_path = os.path.join(os.path.dirname(__file__), "L0_KA")
    if not os.path.exists(dir_path):
        return []
    
    for filename in os.listdir(dir_path):
        if filename.endswith(".html"):
            file_path = os.path.join(dir_path, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # Extract meta id
                meta_id_match = re.search(r'<meta\s+name=["\']id["\']\s+content=["\']([^"\']+)["\']', content, re.IGNORECASE)
                meta_id = meta_id_match.group(1) if meta_id_match else os.path.splitext(filename)[0]
                
                # Extract meta keywords
                meta_kws_match = re.search(r'<meta\s+name=["\']keywords["\']\s+content=["\']([^"\']+)["\']', content, re.IGNORECASE)
                keywords_str = meta_kws_match.group(1) if meta_kws_match else ""
                keywords = [kw.strip() for kw in keywords_str.split(",") if kw.strip()]
                
                # Extract title
                title_match = re.search(r'<title>(.*?)</title>', content, re.IGNORECASE)
                title = title_match.group(1) if title_match else meta_id.replace("_", " ").title()
                
                # Extract <body> content
                body_match = re.search(r'<body>(.*?)</body>', content, re.DOTALL | re.IGNORECASE)
                body_content = body_match.group(1).strip() if body_match else content.strip()
                
                # Translate simple HTML tags to clean Markdown to save Vertex AI tokens
                clean_text = body_content
                clean_text = re.sub(r'<h1[^>]*>(.*?)</h1>', r'# \1\n', clean_text, flags=re.IGNORECASE)
                clean_text = re.sub(r'<h2[^>]*>(.*?)</h2>', r'## \1\n', clean_text, flags=re.IGNORECASE)
                clean_text = re.sub(r'<h3[^>]*>(.*?)</h3>', r'### \1\n', clean_text, flags=re.IGNORECASE)
                clean_text = re.sub(r'<li[^>]*>(.*?)</li>', r'- \1', clean_text, flags=re.IGNORECASE)
                clean_text = re.sub(r'<p[^>]*>(.*?)</p>', r'\1\n', clean_text, flags=re.IGNORECASE | re.DOTALL)
                clean_text = re.sub(r'<br\s*/?>', r'\n', clean_text, flags=re.IGNORECASE)
                clean_text = re.sub(r'<code[^>]*>(.*?)</code>', r'`\1`', clean_text, flags=re.IGNORECASE)
                clean_text = re.sub(r'<strong[^>]*>(.*?)</strong>', r'**\1**', clean_text, flags=re.IGNORECASE)
                clean_text = re.sub(r'<em[^>]*>(.*?)</em>', r'*\1*', clean_text, flags=re.IGNORECASE)
                clean_text = re.sub(r'<[^>]+>', '', clean_text)
                
                # Decode basic html entities
                clean_text = clean_text.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"')
                clean_text = re.sub(r'\n\s*\n', '\n\n', clean_text).strip()
                
                playbooks.append({
                    "id": meta_id,
                    "keywords": keywords,
                    "title": title,
                    "content": clean_text
                })
            except Exception as e:
                # Silently log errors
                pass
                
    return playbooks


def _strip_gs_prefix(value: str) -> str:
    """Normalizes a GCS bucket identifier: strips leading 'gs://' if present.
    Accepts both 'gs://my-bucket' (Cloud Run style) and 'my-bucket' (bare name).
    """
    return value.removeprefix("gs://")


L0_BUCKET    = _strip_gs_prefix(os.environ.get("KNOWLEDGE_STORAGE_BUCKET", "chipllm-l0-playbooks"))
ADHOC_BUCKET = _strip_gs_prefix(os.environ.get("KNOWLEDGE_ADHOC_BUCKET",   "chipllm-adhoc-playbooks"))


def get_file_list(bucket_name: str) -> list[str]:
    """
    Returns the live list of HTML blob names in the given GCS bucket.
    Always performs a fresh list_blobs() call — never cached.
    Callers use this to build a live manifest for diffing against cached content.
    """
    try:
        client = storage.Client()
        return [
            blob.name
            for blob in client.list_blobs(bucket_name)
            if blob.name.endswith(".html")
        ]
    except Exception as exc:
        _log.warning("get_file_list failed for bucket '%s': %s", bucket_name, exc)
        return []


def fetch_file_content(bucket_name: str, blob_name: str) -> dict | None:
    """
    Downloads and parses a single HTML playbook blob.
    Returns a structured playbook dict, or None on any error.
    NOTE: This function is intentionally cache-free. Callers in app.py
    wrap it with @st.cache_data so caching stays in the Streamlit layer.
    """
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        html_text = blob.download_as_text()
        soup = BeautifulSoup(html_text, "html.parser")

        meta_id = soup.find("meta", attrs={"name": "id"})
        meta_keywords = soup.find("meta", attrs={"name": "keywords"})
        body_text = soup.body.get_text(separator=" ", strip=True) if soup.body else ""

        record_id = meta_id["content"] if meta_id else blob_name
        keywords_list = (
            [k.strip().lower() for k in meta_keywords["content"].split(",")]
            if meta_keywords else []
        )
        title_text = (
            soup.title.string.strip()
            if (soup.title and soup.title.string)
            else record_id.replace("_", " ").title()
        )

        return {
            "id": record_id,
            "keywords": keywords_list,
            "title": title_text,
            "content": body_text,
        }
    except Exception as exc:
        _log.warning("fetch_file_content failed for gs://%s/%s: %s", bucket_name, blob_name, exc)
        return None


def build_playbook_pool(
    l0_names: list[str],
    adhoc_names: list[str],
    fetch_fn,
) -> list[dict]:
    """
    Builds the merged playbook list from pre-filtered live manifests.
    `fetch_fn(bucket_name, blob_name)` is a callable — in production this
    will be the @st.cache_data-wrapped version supplied by app.py.

    Merge rules:
    - L0 baseline entries are loaded first.
    - Ad-hoc entries overwrite any matching record_id.
    - Blobs not present in the live manifest are simply never fetched,
      which effectively evicts deleted files from the active pool.
    """
    parsed_ledger: dict[str, dict] = {}

    for blob_name in l0_names:
        entry = fetch_fn(L0_BUCKET, blob_name)
        if entry:
            entry = dict(entry, source="l0_baseline")
            parsed_ledger.setdefault(entry["id"], entry)

    for blob_name in adhoc_names:
        entry = fetch_fn(ADHOC_BUCKET, blob_name)
        if entry:
            entry = dict(entry, source="adhoc")
            parsed_ledger[entry["id"]] = entry  # always overwrite

    return list(parsed_ledger.values())


def load_and_merge_cloud_knowledge_base() -> list[dict]:
    """
    Synchronous (non-cached) initializer used at module load time and as a
    fallback. Caching is NOT applied here — app.py owns that responsibility.
    """
    pool = build_playbook_pool(
        l0_names=get_file_list(L0_BUCKET),
        adhoc_names=get_file_list(ADHOC_BUCKET),
        fetch_fn=fetch_file_content,
    )

    if not pool:
        local_playbooks = _load_playbooks_from_html()
        for p in local_playbooks:
            p["source"] = "local_fallback"
        pool = local_playbooks

    if not pool:
        pool = [dict(p, source="hardcoded_fallback") for p in FALLBACK_PLAYBOOKS]

    return pool


# Grounding runtime context vector array initialization
PLAYBOOKS = load_and_merge_cloud_knowledge_base()
playbooks_pool = PLAYBOOKS


# ---------------------------------------------------------------------------
# RAG retrieval – lightweight keyword matching
# ---------------------------------------------------------------------------

ABBREVIATIONS = {
    "wst": "waystation",
    "waistation": "waystation",
    "kds": "kitchen display system",
    "kvs": "kitchen video system",
    "pos": "point of sale",
    "bos": "back office system",
}

def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
        
    return previous_row[-1]

def fuzzy_match_ratio(s1: str, s2: str) -> float:
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    dist = levenshtein_distance(s1, s2)
    return 1.0 - (dist / max_len)

def preprocess_text(text: str) -> list[str]:
    # Lowercase and clean
    cleaned = text.lower().strip()
    words = re.findall(r'\b\w+\b', cleaned)
    resolved_words = []
    for w in words:
        resolved_words.append(ABBREVIATIONS.get(w, w))
    return resolved_words

def retrieve_context(user_message: str, top_k: int = 1) -> list[dict]:
    """
    Perform fuzzy matching retrieval against the playbook knowledge base.
    Handles typos, abbreviations, and implements a two-pass context builder.
    Ad-hoc playbooks bypass truncation limits and sit at the top.
    """
    query_words = preprocess_text(user_message)
    if not query_words:
        return []
        
    STOPWORDS = {"my", "is", "the", "a", "an", "on", "of", "to", "in", "at", "for", "with", "and", "or", "having", "it", "are", "you"}
    query_clean = " ".join(query_words)
    scored: list[tuple[float, dict]] = []

    for playbook in PLAYBOOKS:
        # Build candidate target strings for this playbook
        candidates = []
        if playbook.get("id"):
            candidates.append(playbook["id"].lower().replace("_", " "))
        if playbook.get("title"):
            candidates.append(playbook["title"].lower())
        for kw in playbook.get("keywords", []):
            candidates.append(kw.lower())

        best_score = 0.0
        
        # Word-by-word fuzzy matching
        for q_word in query_words:
            if len(q_word) < 3 or q_word in STOPWORDS:
                continue
            for cand in candidates:
                cand_words = re.findall(r'\b\w+\b', cand)
                for c_word in cand_words:
                    ratio = fuzzy_match_ratio(q_word, c_word)
                    if ratio > best_score:
                        best_score = ratio

        # Full phrase matching against candidate targets
        for cand in candidates:
            # Check exact substring first
            if query_clean in cand or cand in query_clean:
                # Normalise score between 0.0 and 1.0 (min_len / max_len)
                score = min(len(query_clean), len(cand)) / max(len(query_clean), len(cand)) if max(len(query_clean), len(cand)) > 0 else 0.0
                # Give it a baseline match confidence of 0.6 if it matches substring
                score = max(0.6, score)
                if score > best_score:
                    best_score = score
            ratio = fuzzy_match_ratio(query_clean, cand)
            if ratio > best_score:
                best_score = ratio

        # Threshold of 0.6 to count as a fuzzy match
        if best_score >= 0.6:
            scored.append((best_score, playbook))

    # Sort matches by score descending
    scored.sort(key=lambda x: x[0], reverse=True)
    matched_playbooks = [p for _, p in scored]

    # Pass 2: Separate adhoc and baseline matches
    adhoc_matches = [p for p in matched_playbooks if p.get("source") == "adhoc"]
    baseline_matches = [p for p in matched_playbooks if p.get("source") != "adhoc"]

    # Ad-hoc matches bypass length truncation and sit at the top.
    # Baseline matches are limited to top_k.
    returned_playbooks = adhoc_matches + baseline_matches[:top_k]
    return returned_playbooks

