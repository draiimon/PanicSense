-- Import data from test-data-variant.csv
CREATE TEMP TABLE temp_variant (
  message TEXT,
  date TIMESTAMP,
  platform TEXT,
  place TEXT,
  sentiment TEXT,
  event_type TEXT
);

-- Note: Adjust path if necessary
COPY temp_variant FROM '/home/runner/workspace/test-data-variant.csv' DELIMITER ',' CSV HEADER;

-- Insert data into sentiment_posts table
INSERT INTO sentiment_posts (text, timestamp, source, language, sentiment, confidence, location, "disasterType")
SELECT 
  message,
  date,
  platform,
  'en', -- default language
  sentiment,
  0.85, -- default confidence
  place,
  event_type
FROM temp_variant;

-- Import data from test-data-with-location.csv
CREATE TEMP TABLE temp_location (
  "Text" TEXT,
  "Timestamp" TIMESTAMP,
  "Source" TEXT,
  "Location" TEXT,
  "Disaster" TEXT
);

-- Note: Adjust path if necessary
COPY temp_location FROM '/home/runner/workspace/test-data-with-location.csv' DELIMITER ',' CSV HEADER;

-- Insert data into sentiment_posts table
INSERT INTO sentiment_posts (text, timestamp, source, language, sentiment, confidence, location, "disasterType")
SELECT 
  "Text",
  "Timestamp",
  "Source",
  CASE WHEN "Text" ~ '(ang|ng|mga|ako|hindi|sobrang|talaga)' THEN 'tl' ELSE 'en' END, -- Simple language detection
  CASE
    WHEN "Text" ~ '(takot|nakakatakot|kabado|kinakabahan)' THEN 'Fear/Anxiety'
    WHEN "Text" ~ '(galit|inis|yamot)' THEN 'Anger'
    WHEN "Text" ~ '(malungkot|lungkot)' THEN 'Sadness'
    WHEN "Text" ~ '(tulong|saklolo|tabang|delikado)' THEN 'Panic'
    WHEN "Text" ~ '(pag-asa|umaasa)' THEN 'Hope'
    WHEN "Text" ~ '(salamat|ginhawa|ligtas|nakaligtas)' THEN 'Relief'
    WHEN "Text" ~ '(😭|😢)' THEN 'Sadness'
    WHEN "Text" ~ '(😱|😨)' THEN 'Fear/Anxiety'
    WHEN "Text" ~ '(😡)' THEN 'Anger'
    WHEN "Text" ~ '(🙏)' THEN 'Hope'
    ELSE 'Neutral'
  END,
  0.80, -- default confidence
  "Location",
  "Disaster"
FROM temp_location;

-- Import data from test-data.csv
CREATE TEMP TABLE temp_data (
  message TEXT,
  date TIMESTAMP, 
  platform TEXT,
  sentiment TEXT,
  event_type TEXT
);

-- Note: Adjust path if necessary
COPY temp_data FROM '/home/runner/workspace/test-data.csv' DELIMITER ',' CSV HEADER;

-- Insert data into sentiment_posts table
INSERT INTO sentiment_posts (text, timestamp, source, language, sentiment, confidence, location, "disasterType")
SELECT 
  message,
  date,
  platform,
  'en', -- default language
  sentiment,
  0.82, -- default confidence
  NULL, -- location not available
  event_type
FROM temp_data;

-- Import data from sample-sentiments.csv
CREATE TEMP TABLE temp_samples (
  text TEXT,
  timestamp TIMESTAMP,
  source TEXT,
  sentiment TEXT
);

-- Note: Adjust path if necessary
COPY temp_samples FROM '/home/runner/workspace/sample-sentiments.csv' DELIMITER ',' CSV HEADER;

-- Insert data into sentiment_posts table
INSERT INTO sentiment_posts (text, timestamp, source, language, sentiment, confidence, location, "disasterType")
SELECT 
  text,
  timestamp,
  source,
  'en', -- default language
  sentiment,
  0.82, -- default confidence
  NULL, -- location not available
  NULL  -- disaster type not available
FROM temp_samples;

-- Create disaster events based on unique disaster types
INSERT INTO disaster_events (name, description, timestamp, location, type, "sentimentImpact")
SELECT 
  DISTINCT ON (sp."disasterType") sp."disasterType" || ' Event',
  'A ' || LOWER(sp."disasterType") || ' event with multiple posts',
  MIN(sp.timestamp),
  MAX(sp.location),
  sp."disasterType",
  (
    SELECT s2.sentiment
    FROM sentiment_posts s2
    WHERE s2."disasterType" = sp."disasterType"
    GROUP BY s2.sentiment
    ORDER BY COUNT(*) DESC
    LIMIT 1
  )
FROM sentiment_posts sp
WHERE sp."disasterType" IS NOT NULL
GROUP BY sp."disasterType";

-- Cleanup
DROP TABLE temp_variant;
DROP TABLE temp_location;
DROP TABLE temp_data;
DROP TABLE temp_samples;