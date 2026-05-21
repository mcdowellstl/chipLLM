import json
import os
import re

def main():
    # Load diagnostic graph
    with open("diagnostic-graph.json", "r") as f:
        graph = json.load(f)
        
    nodes = graph["nodes"]
    
    # Create L0_KA directory if not exists
    os.makedirs("L0_KA", exist_ok=True)
    
    # Let's count how many playbooks we generate
    generated_count = 0
    
    # Define a helper to generate keywords based on the node context and category
    def get_keywords(node_id, title, subtitle):
        # Base keywords based on prefix
        kws = []
        if node_id.startswith("printer_"):
            kws.extend(["printer", "print", "receipt", "paper", "feed", "epson", "star", "bixolon"])
        elif node_id.startswith("pos_"):
            kws.extend(["pos", "terminal", "register", "cashier", "checkout", "screen", "elo", "terminal"])
        elif node_id.startswith("kds_"):
            kws.extend(["kds", "kitchen", "display", "screen", "monitor", "order", "makeline"])
        elif node_id.startswith("kiosk_"):
            kws.extend(["kiosk", "self-order", "customer", "tillster", "acrelec"])
        elif node_id.startswith("other_"):
            kws.extend(["network", "wifi", "router", "drawer", "cash drawer", "switch", "ap"])
            
        # Add words from title & subtitle (lowercase, alphanumeric only, length > 3)
        combined = f"{title} {subtitle}"
        words = re.findall(r"\b\w{3,}\b", combined.lower())
        kws.extend(words)
        
        # Unique list
        unique_kws = []
        for kw in kws:
            if kw not in unique_kws:
                unique_kws.append(kw)
        return ", ".join(unique_kws[:15])  # Cap at 15 keywords
        
    for node_id, node in nodes.items():
        # Skip terminal nodes as they are endpoints, not playbooks
        if node["type"] in ["terminal_resolved", "terminal_submit_ticket"]:
            continue
            
        title = node.get("title", "Troubleshooting Guide")
        subtitle = node.get("subtitle", "")
        node_type = node["type"]
        
        # Build keywords
        keywords = get_keywords(node_id, title, subtitle)
        
        # Build body content based on node type
        body_html = f"<h1>{title}</h1>\n"
        if subtitle:
            body_html += f"<p class='subtitle'><em>{subtitle}</em></p>\n"
            
        body_html += "<div class='playbook-details'>\n"
        
        if node_type == "quick_fix":
            body_html += "  <h2>Step-by-Step Instructions</h2>\n"
            body_html += "  <ol>\n"
            for inst in node.get("instructions", []):
                body_html += f"    <li>{inst}</li>\n"
            body_html += "  </ol>\n"
            body_html += f"  <p class='resolved-prompt'><strong>Verification Question:</strong> {node.get('resolvedPrompt', 'Did this resolve the issue?')}</p>\n"
            
        elif node_type == "multiple_choice":
            body_html += "  <h2>Triage Options / Symptoms</h2>\n"
            body_html += "  <p>Please check the symptoms matching the issue:</p>\n"
            body_html += "  <ul>\n"
            for opt in node.get("options", []):
                body_html += f"    <li><strong>{opt['label']}</strong> (routes to: <code>{opt['next']}</code>)</li>\n"
            body_html += "  </ul>\n"
            
        elif node_type == "yes_no":
            body_html += "  <h2>Diagnostic Check</h2>\n"
            body_html += f"  <p>Is this statement true? <strong>{title}</strong></p>\n"
            body_html += "  <ul>\n"
            body_html += f"    <li><strong>YES</strong> -> routes to: <code>{node.get('yes')}</code></li>\n"
            body_html += f"    <li><strong>NO</strong> -> routes to: <code>{node.get('no')}</code></li>\n"
            body_html += "  </ul>\n"
            
        elif node_type == "text_input" or node_type == "photo_upload":
            body_html += "  <h2>Information Gathering Step</h2>\n"
            prompt = node.get("prompt", "Enter requested information.")
            body_html += f"  <p>{prompt}</p>\n"
            if "fieldKey" in node:
                body_html += f"  <p>Saves to field: <code>{node['fieldKey']}</code></p>\n"
                
        elif node_type == "multi_field_optional":
            body_html += "  <h2>Device Details Collection</h2>\n"
            body_html += "  <p>The following fields are optional but recommended to facilitate faster support:</p>\n"
            body_html += "  <ul>\n"
            for fld in node.get("fields", []):
                body_html += f"    <li>{fld['label']} (type: {fld['type']})</li>\n"
            body_html += "  </ul>\n"
            
        body_html += "</div>\n"
        
        # Wrap in full HTML document
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <meta name="id" content="{node_id}">
    <meta name="keywords" content="{keywords}">
    <style>
        body {{
            font-family: 'Inter', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1 {{
            color: #ea580c;
            border-bottom: 2px solid #f97316;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #1e3a5f;
            margin-top: 20px;
        }}
        .subtitle {{
            color: #666;
            font-size: 1.1em;
        }}
        ol, ul {{
            padding-left: 20px;
        }}
        li {{
            margin-bottom: 8px;
        }}
        code {{
            background-color: #f1f5f9;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: monospace;
        }}
        .playbook-details {{
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 15px 20px;
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    {body_html}
</body>
</html>
"""
        # Write to file
        file_path = os.path.join("L0_KA", f"{node_id}.html")
        with open(file_path, "w", encoding="utf-8") as out_f:
            out_f.write(html_content)
        generated_count += 1
        print(f"Generated {file_path}")
        
    print(f"Successfully generated {generated_count} HTML playbooks in L0_KA/")

if __name__ == "__main__":
    main()
