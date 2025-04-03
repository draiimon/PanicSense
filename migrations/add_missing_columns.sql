-- Check if ai_trust_message column exists in sentiment_posts
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'sentiment_posts' AND column_name = 'ai_trust_message'
    ) THEN
        ALTER TABLE sentiment_posts ADD COLUMN ai_trust_message text;
    END IF;
END $$;

-- Check if sentiment_feedback table exists and add columns if needed
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.tables 
        WHERE table_name = 'sentiment_feedback'
    ) THEN
        -- Check for missing columns in sentiment_feedback
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns 
            WHERE table_name = 'sentiment_feedback' AND column_name = 'ai_trust_message'
        ) THEN
            ALTER TABLE sentiment_feedback ADD COLUMN ai_trust_message text;
        END IF;
        
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns 
            WHERE table_name = 'sentiment_feedback' AND column_name = 'possible_trolling'
        ) THEN
            ALTER TABLE sentiment_feedback ADD COLUMN possible_trolling boolean DEFAULT false;
        END IF;
    ELSE
        -- Create sentiment_feedback table if it doesn't exist
        CREATE TABLE sentiment_feedback (
            id serial PRIMARY KEY NOT NULL,
            original_post_id integer REFERENCES sentiment_posts(id) ON DELETE CASCADE,
            original_text text NOT NULL,
            original_sentiment text NOT NULL,
            corrected_sentiment text DEFAULT '',
            corrected_location text,
            corrected_disaster_type text,
            trained_on boolean DEFAULT false,
            created_at timestamp DEFAULT now(),
            user_id integer REFERENCES users(id),
            ai_trust_message text,
            possible_trolling boolean DEFAULT false
        );
    END IF;
END $$;

-- Create training_examples table if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.tables 
        WHERE table_name = 'training_examples'
    ) THEN
        CREATE TABLE training_examples (
            id serial PRIMARY KEY NOT NULL,
            text text NOT NULL,
            text_key text NOT NULL,
            sentiment text NOT NULL,
            language text NOT NULL,
            confidence real NOT NULL DEFAULT 0.95,
            created_at timestamp DEFAULT now(),
            updated_at timestamp DEFAULT now(),
            CONSTRAINT training_examples_text_unique UNIQUE(text),
            CONSTRAINT training_examples_text_key_unique UNIQUE(text_key)
        );
    END IF;
END $$;