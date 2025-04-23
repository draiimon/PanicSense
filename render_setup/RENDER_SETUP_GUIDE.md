# Render Deployment Guide for PanicSense

## Prerequisites

1. GitHub repository with your PanicSense code
2. Render account
3. PostgreSQL database (Render or external)

## Deployment Steps

### Step 1: Setup Database

1. Create a PostgreSQL database in Render or use an external one
2. Save the DATABASE_URL for later use

### Step 2: Configure Render Web Service

1. Go to the Render dashboard
2. Click "New" and select "Web Service"
3. Connect your GitHub repository
4. Configure the service with the following settings:

```
Name: PanicSense
Region: Singapore (Southeast Asia)
Branch: main
Root Directory: (leave blank)
Build Command: bash ./render_setup/build_command.sh
Start Command: npm run start
```

### Step 3: Add Environment Variables

Navigate to the "Environment" tab and add the following variables:

```
DATABASE_URL=<your PostgreSQL database URL>
NODE_ENV=production
API_KEY_1=<your Groq API key>
```

Add any additional environment variables needed for your project.

### Step 4: Deploy

1. Click "Create Web Service"
2. Wait for the deployment to complete
3. Access your application at the provided URL

## Troubleshooting

If you encounter issues during deployment:

1. Check the build and runtime logs in Render dashboard
2. Verify that the requirements.txt file is correctly installed
3. Ensure the Python files are accessible in both python/ and server/python/ directories
4. Check that the environment variables are correctly set
5. Verify the database connection

## File Structure Requirements

PanicSense requires the following file structure to work properly:

```
/
├── python/
│   ├── process.py          # Main Python processing code
│   └── emoji_utils.py      # Emoji utilities for text processing
├── server/
│   └── python/
│       ├── process.py      # Duplicate for server access
│       └── emoji_utils.py  # Duplicate for server access
├── requirements.txt        # Python dependencies
├── package.json            # Node.js dependencies and scripts
└── ...                     # Other application files
```

The build command will ensure the proper file structure is maintained during deployment.