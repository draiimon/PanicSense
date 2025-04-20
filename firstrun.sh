#!/bin/bash

# This script checks if the application is being run for the first time
# If so, it will run the setup script

# Set colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if .env file exists
if [ ! -f .env ]; then
    echo -e "\n${BLUE}===============================================${NC}"
    echo -e "${BLUE}     WELCOME TO PANICSENSE PH FIRST SETUP       ${NC}"
    echo -e "${BLUE}===============================================${NC}"
    echo -e "\nDetected first run of PanicSense PH!"
    echo -e "Running setup script to configure your environment...\n"
    
    # Run the setup script
    node setup.js
    
    # Create a marker file to indicate setup has run
    touch .setup_complete
    
    echo -e "\n${GREEN}First-time setup completed!${NC}"
    echo -e "You can run the setup script again at any time with: ${YELLOW}node setup.js${NC}\n"
else
    # Check if this is not the first run but setup hasn't completed
    if [ ! -f .setup_complete ] && [ -f .env ]; then
        echo -e "\n${YELLOW}It looks like you have a .env file but setup may not have completed.${NC}"
        echo -e "If you need to run the setup again, use: ${YELLOW}node setup.js${NC}\n"
        
        # Create the marker anyway
        touch .setup_complete
    fi
fi