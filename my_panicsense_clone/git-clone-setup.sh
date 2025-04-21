#!/bin/bash

# This script helps users automatically set up the git hooks
# to run firstrun.sh after cloning the repository

# Set colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "\n${BLUE}=================================================${NC}"
echo -e "${BLUE}    PANICSENSE PH GIT HOOKS SETUP UTILITY        ${NC}"
echo -e "${BLUE}=================================================${NC}"

# Check if we're in a git repository
if [ ! -d .git ]; then
    echo -e "\n${RED}Error: This script must be run from the root of the git repository.${NC}"
    echo -e "Please navigate to the project root directory and try again.\n"
    exit 1
fi

# Create git hooks directory if it doesn't exist
mkdir -p .git/hooks

# Create post-checkout hook
POST_CHECKOUT_HOOK=".git/hooks/post-checkout"

echo -e "\n${YELLOW}Setting up Git post-checkout hook...${NC}"

cat > "$POST_CHECKOUT_HOOK" << 'EOF'
#!/bin/bash

# This hook runs after a successful git checkout
# We use it to automatically run firstrun.sh for new clones

# Only run this after the initial clone 
# (when HEAD_NEW is equal to the initial commit)
if [ -f .setup_complete ]; then
    # Setup already ran before, don't run again automatically
    exit 0
fi

# Run firstrun.sh if it exists
if [ -f ./firstrun.sh ]; then
    echo "Running first-time setup script..."
    chmod +x ./firstrun.sh
    ./firstrun.sh
fi
EOF

# Make the hook executable
chmod +x "$POST_CHECKOUT_HOOK"

echo -e "${GREEN}✓ Git hook setup completed successfully!${NC}"
echo -e "The firstrun.sh script will automatically run after new repository clones."
echo -e "\n${YELLOW}Note for new users:${NC} After cloning this repository, the setup wizard will run automatically."
echo -e "You can also run it manually with ${YELLOW}./firstrun.sh${NC} at any time.\n"