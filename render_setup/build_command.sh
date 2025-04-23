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