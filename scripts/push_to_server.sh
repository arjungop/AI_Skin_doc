#!/bin/bash
# Push code to A100 server
# Syncs only code, not datasets

# Configure these
SERVER_USER="your_username"
SERVER_HOST="your.server.edu"
SERVER_PATH="~/Skin-Doc"

echo "Pushing code to server"

if [ "$SERVER_USER" = "your_username" ]; then
    echo "Please configure your server details in this script first:"
    echo "  Edit: scripts/push_to_server.sh"
    echo "  Set: SERVER_USER, SERVER_HOST, SERVER_PATH"
    read -p "Enter server user: " SERVER_USER
    read -p "Enter server host: " SERVER_HOST
    read -p "Enter server path [~/Skin-Doc]: " input_path
    SERVER_PATH=${input_path:-"~/Skin-Doc"}
fi

echo "Pushing to: ${SERVER_USER}@${SERVER_HOST}:${SERVER_PATH}"
echo

# Create remote directory (if not already exists)
echo "[1/3] Checking remote directory..."
ssh ${SERVER_USER}@${SERVER_HOST} "mkdir -p ${SERVER_PATH}"
echo "[OK] Directory ready (created or already exists)"

# Push code (excluding large files)
echo "[2/3] Syncing code files..."
rsync -avz --progress \
    --exclude-from='.serverignore' \
    --exclude='.git' \
    ./ ${SERVER_USER}@${SERVER_HOST}:${SERVER_PATH}/

# Make scripts executable
echo "[3/3] Setting permissions..."
ssh ${SERVER_USER}@${SERVER_HOST} "chmod +x ${SERVER_PATH}/scripts/*.sh"

echo
echo -e "${GREEN}✅ Push complete!${NC}"
echo
echo "Next steps on server:"
echo "  ssh ${SERVER_USER}@${SERVER_HOST}"
echo "  cd ${SERVER_PATH}"
echo "  bash scripts/server_setup.sh"
