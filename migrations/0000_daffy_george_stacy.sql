CREATE TABLE "analyzed_files" (
	"id" serial PRIMARY KEY NOT NULL,
	"original_name" text NOT NULL,
	"stored_name" text NOT NULL,
	"timestamp" timestamp DEFAULT now() NOT NULL,
	"record_count" integer NOT NULL,
	"evaluation_metrics" json,
	"uploaded_by" integer
);
--> statement-breakpoint
CREATE TABLE "disaster_events" (
	"id" serial PRIMARY KEY NOT NULL,
	"name" text NOT NULL,
	"description" text,
	"timestamp" timestamp DEFAULT now() NOT NULL,
	"location" text,
	"type" text NOT NULL,
	"sentiment_impact" text,
	"created_by" integer
);
--> statement-breakpoint
CREATE TABLE "profile_images" (
	"id" serial PRIMARY KEY NOT NULL,
	"name" text NOT NULL,
	"role" text NOT NULL,
	"image_url" text NOT NULL,
	"description" text,
	"created_at" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "sentiment_posts" (
	"id" serial PRIMARY KEY NOT NULL,
	"text" text NOT NULL,
	"timestamp" timestamp DEFAULT now() NOT NULL,
	"source" text,
	"language" text,
	"sentiment" text NOT NULL,
	"confidence" real NOT NULL,
	"location" text,
	"disaster_type" text,
	"file_id" integer,
	"explanation" text,
	"processed_by" integer
);
--> statement-breakpoint
CREATE TABLE "sessions" (
	"id" serial PRIMARY KEY NOT NULL,
	"user_id" integer NOT NULL,
	"token" text NOT NULL,
	"created_at" timestamp DEFAULT now() NOT NULL,
	"expires_at" timestamp NOT NULL,
	CONSTRAINT "sessions_token_unique" UNIQUE("token")
);
--> statement-breakpoint
CREATE TABLE "users" (
	"id" serial PRIMARY KEY NOT NULL,
	"username" text NOT NULL,
	"password" text NOT NULL,
	"email" text NOT NULL,
	"full_name" text NOT NULL,
	"created_at" timestamp DEFAULT now() NOT NULL,
	"role" text DEFAULT 'user' NOT NULL,
	CONSTRAINT "users_username_unique" UNIQUE("username"),
	CONSTRAINT "users_email_unique" UNIQUE("email")
);
--> statement-breakpoint
ALTER TABLE "analyzed_files" ADD CONSTRAINT "analyzed_files_uploaded_by_users_id_fk" FOREIGN KEY ("uploaded_by") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "disaster_events" ADD CONSTRAINT "disaster_events_created_by_users_id_fk" FOREIGN KEY ("created_by") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "sentiment_posts" ADD CONSTRAINT "sentiment_posts_processed_by_users_id_fk" FOREIGN KEY ("processed_by") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "sessions" ADD CONSTRAINT "sessions_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;