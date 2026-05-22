-- 1. Create a sequence for the numeric part of the identifier (starting at 1027 after mock data)
CREATE SEQUENCE IF NOT EXISTS cases_id_seq START WITH 1027;

-- 2. Create a PL/pgSQL trigger function to format the ID automatically on insert
CREATE OR REPLACE FUNCTION generate_case_id()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.id IS NULL THEN
        NEW.id := 'RC' || lpad(nextval('cases_id_seq')::text, 6, '0');
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 3. Create the cases table (production schema)
CREATE TABLE IF NOT EXISTS public.cases (
    id                CHARACTER VARYING(20)  NOT NULL,
    store_id          CHARACTER VARYING(50)  NOT NULL,
    description       TEXT                   NOT NULL,
    status            CHARACTER VARYING(50)  NOT NULL DEFAULT 'New',
    comments          JSONB                  NOT NULL DEFAULT '[]'::jsonb,
    created_at        TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    category          CHARACTER VARYING(50)  NULL,
    subcategory       CHARACTER VARYING(50)  NULL,
    priority          SMALLINT               NULL,
    escalations       SMALLINT               NOT NULL DEFAULT 0,
    short_description TEXT                   NULL,
    CONSTRAINT cases_pkey PRIMARY KEY (id)
) TABLESPACE pg_default;

-- 4. Bind trigger to the cases table
CREATE OR REPLACE TRIGGER trigger_generate_case_id
BEFORE INSERT ON cases
FOR EACH ROW
EXECUTE FUNCTION generate_case_id();

-- 5. Seed the initial mock cases
INSERT INTO public.cases (id, store_id, description, short_description, status, priority, escalations, category, subcategory, comments, created_at)
VALUES
(
    'RC001024', '67067',
    'Kiosk 3 Cash Acceptor Jammed',
    'Kiosk 3 Cash Acceptor Jammed',
    'Pending', 2, 0, 'hardware', 'kiosk',
    '[{"timestamp": "2026-05-21 10:15:00", "author": "System", "text": "Incident opened automatically by device heartbeat alert."}, {"timestamp": "2026-05-21 11:30:00", "author": "Tech Support", "text": "Dispatching local tech John to site. ETA is 2 hours."}]'::jsonb,
    '2026-05-21 14:15:00+00'
),
(
    'RC001025', '67067',
    'KVS Bumpbar buttons unresponsive in kitchen Zone 1',
    'KVS Bumpbar buttons unresponsive in kitchen Zone 1',
    'In Progress', 3, 0, 'hardware', 'bumpbar',
    '[{"timestamp": "2026-05-21 13:45:00", "author": "Manager Jim", "text": "Kitchen staff reports keys 3 and 4 are not registering when pressed."}, {"timestamp": "2026-05-21 14:00:00", "author": "Tech Support", "text": "Rebooted KVS device remotely. Issue persists."}]'::jsonb,
    '2026-05-21 17:45:00+00'
),
(
    'RC001026', '67067',
    'POS 2 receipt printer paper jam sensor failure',
    'POS 2 receipt printer paper jam sensor failure',
    'New', 3, 0, 'hardware', 'printer',
    '[{"timestamp": "2026-05-21 15:10:00", "author": "System", "text": "Jam detected in sensor corridor. User cleared jam but sensor remains flagged red."}]'::jsonb,
    '2026-05-21 19:10:00+00'
),
(
    'RC003840', '67068',
    'AC unit in dining area blowing warm air',
    'AC unit in dining area blowing warm air',
    'Open', 4, 0, 'hardware', 'bos',
    '[]'::jsonb,
    '2026-05-21 20:00:00+00'
)
ON CONFLICT (id) DO NOTHING;
