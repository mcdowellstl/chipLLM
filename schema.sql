-- 1. Create a sequence for the numeric part of the identifier (starting at 1027 after mock data)
CREATE SEQUENCE IF NOT EXISTS cases_id_seq START WITH 1027;

-- 2. Create the cases table
CREATE TABLE IF NOT EXISTS cases (
    id VARCHAR(20) PRIMARY KEY,
    store_id VARCHAR(50) NOT NULL,
    summary TEXT NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'New',
    comments JSONB NOT NULL DEFAULT '[]'::jsonb,
    category VARCHAR(50),
    subcategory VARCHAR(50),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 3. Create a PL/pgSQL trigger function to format the ID automatically on insert
CREATE OR REPLACE FUNCTION generate_case_id()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.id IS NULL THEN
        NEW.id := 'RC' || lpad(nextval('cases_id_seq')::text, 6, '0');
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 4. Bind trigger to the cases table
CREATE OR REPLACE TRIGGER trigger_generate_case_id
BEFORE INSERT ON cases
FOR EACH ROW
EXECUTE FUNCTION generate_case_id();

-- 5. Seed the initial mock cases
INSERT INTO cases (id, store_id, summary, status, category, subcategory, comments, created_at)
VALUES
('RC001024', '67067', 'Kiosk 3 Cash Acceptor Jammed', 'Assigned to Field Tech', 'hardware', 'kiosk',
  '[{"timestamp": "2026-05-21 10:15:00", "author": "System", "text": "Incident opened automatically by device heartbeat alert."}, {"timestamp": "2026-05-21 11:30:00", "author": "Tech Support", "text": "Dispatching local tech John to site. ETA is 2 hours."}]'::jsonb,
  '2026-05-21 10:15:00-04'),
('RC001025', '67067', 'KVS Bumpbar buttons unresponsive in kitchen Zone 1', 'In Progress', 'hardware', 'kvs',
  '[{"timestamp": "2026-05-21 13:45:00", "author": "Manager Jim", "text": "Kitchen staff reports keys 3 and 4 are not registering when pressed."}, {"timestamp": "2026-05-21 14:00:00", "author": "Tech Support", "text": "Rebooted KVS device remotely. Issue persists."}]'::jsonb,
  '2026-05-21 13:45:00-04'),
('RC001026', '67067', 'POS 2 receipt printer paper jam sensor failure', 'New', 'hardware', 'printer',
  '[{"timestamp": "2026-05-21 15:10:00", "author": "System", "text": "Jam detected in sensor corridor. User cleared jam but sensor remains flagged red."}]'::jsonb,
  '2026-05-21 15:10:00-04'),
('RC003840', '67068', 'AC unit in dining area blowing warm air', 'Open', 'hardware', 'bos',
  '[]'::jsonb,
  '2026-05-21 16:00:00-04')
ON CONFLICT (id) DO NOTHING;
