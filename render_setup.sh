#!/bin/bash
# Render Setup Script for PanicSense
# Run this script to prepare your project for deployment to Render

# Define colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}PanicSense Render Deployment Setup${NC}"
echo -e "${BLUE}========================================${NC}"

# Create render_setup directory
mkdir -p render_setup/python

# Generate requirements.txt
echo -e "${BLUE}Generating requirements.txt file...${NC}"
cat > requirements.txt << 'EOL'
anthropic==0.19.4
beautifulsoup4==4.12.3
langdetect==1.0.9
nltk==3.8.1
numpy==1.26.4
openai==1.16.2
pandas==2.1.4
python-dotenv==1.0.1
pytz==2024.1
requests==2.31.0
scikit-learn==1.3.2
snscrape==0.7.0.20240301
soupsieve==2.5
torch==2.1.2
tqdm==4.66.2
transformers==4.39.3
protobuf==4.25.3
EOL
echo -e "${GREEN}✓ Created requirements.txt${NC}"

# Create build command script
echo -e "${BLUE}Creating build command script...${NC}"
cat > render_setup/build_command.sh << 'EOL'
#!/bin/bash

# This file shows the exact build command to use in Render

echo "=== Preparing environment ==="
if [ ! -f "requirements.txt" ]; then
  echo "Creating requirements.txt from template..."
  cp ./render_setup/requirements.txt ./requirements.txt
fi

echo "=== Installing Python dependencies ==="
pip3 install -r requirements.txt

echo "=== Installing Node.js dependencies ==="
npm install

echo "=== Building React app ==="
npm run build

echo "=== Checking Python directories ==="
# Make sure the python directory structure is correct
if [ ! -d "python" ]; then
  echo "Creating python directory..."
  mkdir -p python
fi

# Make sure process.py and emoji_utils.py are available in the python directory
if [ -f "server/python/process.py" ] && [ ! -f "python/process.py" ]; then
  echo "Copying process.py to python directory..."
  cp server/python/process.py python/process.py
fi

if [ -f "server/python/emoji_utils.py" ] && [ ! -f "python/emoji_utils.py" ]; then
  echo "Copying emoji_utils.py to python directory..."
  cp server/python/emoji_utils.py python/emoji_utils.py
fi

echo "=== Build completed successfully ==="
EOL
chmod +x render_setup/build_command.sh
echo -e "${GREEN}✓ Created build_command.sh${NC}"

# Create Procfile
echo -e "${BLUE}Creating Procfile...${NC}"
cat > Procfile << 'EOL'
web: npm run start
EOL
echo -e "${GREEN}✓ Created Procfile${NC}"

# Create README for the render_setup directory
echo -e "${BLUE}Creating setup documentation...${NC}"
cat > render_setup/README.md << 'EOL'
# Render Setup for PanicSense

This directory contains files needed to configure your PanicSense application for deployment on Render.

## Files

- `build_command.sh`: The build command script to use in your Render configuration
- `requirements.txt`: A backup of the Python requirements in case the root file is missing
- `RENDER_SETUP_GUIDE.md`: A comprehensive guide to deploying on Render

## How to Deploy

See `RENDER_SETUP_GUIDE.md` for detailed deployment instructions.
EOL
echo -e "${GREEN}✓ Created README.md${NC}"

# Backup requirements.txt
cp requirements.txt render_setup/requirements.txt

echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${BLUE}To deploy to Render:${NC}"
echo -e "1. Push these changes to your GitHub repository"
echo -e "2. Connect your repository to Render"
echo -e "3. Follow the instructions in render_setup/RENDER_SETUP_GUIDE.md"
echo -e "${BLUE}========================================${NC}"