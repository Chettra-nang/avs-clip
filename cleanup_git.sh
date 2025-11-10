#!/bin/bash
# Script to remove large files from git history

set -e

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║              Git Repository Cleanup Script                          ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if git-filter-repo is installed
if ! command -v git-filter-repo &> /dev/null; then
    echo "Installing git-filter-repo..."
    pip install git-filter-repo
fi

echo "Current repository size:"
du -sh .git

echo ""
echo "Large files in git history:"
git rev-list --objects --all | \
    git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | \
    awk '/^blob/ {print substr($0,6)}' | \
    sort -k2 -n -r | \
    head -10

echo ""
echo "⚠️  WARNING: This will rewrite git history!"
echo "This will remove:"
echo "  - models/clip_verify/ (605MB model file)"
echo "  - All test data and temporary files"
echo ""
read -p "Continue? (yes/no): " -r
if [[ ! $REPLY =~ ^yes$ ]]; then
    echo "Aborted."
    exit 1
fi

echo ""
echo "Creating backup..."
cp -r .git .git.backup
echo "✓ Backup created at .git.backup"

echo ""
echo "Removing large files from history..."

# Remove the large model file
git filter-repo --path models/clip_verify --invert-paths --force

# Remove test data directories
git filter-repo --path data_test --invert-paths --force
git filter-repo --path data_test_fixed --invert-paths --force
git filter-repo --path test_output --invert-paths --force
git filter-repo --path test_outputs --invert-paths --force

# Remove test files
git filter-repo --path-glob 'test_*.py' --invert-paths --force
git filter-repo --path-glob 'test_*.png' --invert-paths --force
git filter-repo --path-glob 'TASK_*.md' --invert-paths --force
git filter-repo --path-glob '*_SUMMARY.md' --invert-paths --force --use-base-name
git filter-repo --path-glob '*_IMPLEMENTATION.md' --invert-paths --force --use-base-name
git filter-repo --path-glob '*_FIX.md' --invert-paths --force --use-base-name
git filter-repo --path-glob '*_ANALYSIS.md' --invert-paths --force --use-base-name

echo ""
echo "Cleaning up..."
git reflog expire --expire=now --all
git gc --prune=now --aggressive

echo ""
echo "New repository size:"
du -sh .git

echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                    Cleanup Complete!                                 ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "⚠️  IMPORTANT: You need to force push to update the remote:"
echo ""
echo "  git remote add origin <your-repo-url>  # If not already added"
echo "  git push -u origin main --force"
echo ""
echo "⚠️  WARNING: This will overwrite the remote repository!"
echo "   All collaborators will need to re-clone the repository."
echo ""
echo "If something went wrong, restore from backup:"
echo "  rm -rf .git"
echo "  mv .git.backup .git"
echo ""
