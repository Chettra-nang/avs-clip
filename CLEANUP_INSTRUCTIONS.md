# Git Repository Cleanup Instructions

Your git repository is **537MB** because it contains a large model file (`models/clip_verify/model.safetensors` - 605MB) that was accidentally committed.

## Problem

```bash
$ du -sh .git
537M    .git

$ git push
Compressing objects: 100% (46/46), done.
Writing objects:  76% (38/50), 275.83 MiB | 3.39 MiB/s  # STUCK!
```

The large file is in git history and needs to be removed.

## Solution Options

### Option 1: Quick Fix - Start Fresh (Recommended)

If you haven't pushed to remote yet, the easiest solution is to start with a clean repository:

```bash
# 1. Backup your code
cd ~/Desktop
cp -r avs avs_backup

# 2. Remove git history
cd avs
rm -rf .git

# 3. Initialize fresh repository
git init
git add .
git commit -m "Initial commit - VLM-MARL ambulance priority system"

# 4. Check size (should be <10MB)
du -sh .git

# 5. Push to remote
git remote add origin <your-repo-url>
git branch -M main
git push -u origin main --force
```

### Option 2: Clean Git History (Advanced)

If you need to preserve git history, use BFG Repo-Cleaner:

```bash
# 1. Install BFG
sudo apt install default-jre
wget https://repo1.maven.org/maven2/com/madgag/bfg/1.14.0/bfg-1.14.0.jar
alias bfg='java -jar bfg-1.14.0.jar'

# 2. Backup
cp -r .git .git.backup

# 3. Remove large files
bfg --delete-files model.safetensors
bfg --delete-files merges.txt
bfg --delete-folders models/clip_verify

# 4. Clean up
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# 5. Check size
du -sh .git  # Should be <10MB

# 6. Force push
git push origin main --force
```

### Option 3: Use git-filter-repo (Alternative)

```bash
# 1. Install
pip install git-filter-repo

# 2. Backup
cp -r .git .git.backup

# 3. Remove large files
git filter-repo --path models/clip_verify --invert-paths --force

# 4. Clean up
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# 5. Check size
du -sh .git

# 6. Force push
git push origin main --force
```

## Updated .gitignore

The `.gitignore` has been updated to prevent this in the future:

```gitignore
# Models and large files
*.pth
*.pt
*.safetensors
models/clip/
models/clip_*/
models/clip_verify/

# Data
data/
data_*/

# Test files
test_*/
test_*.py
test_*.png
```

## Verification

After cleanup, verify the repository size:

```bash
# Check git size
du -sh .git
# Should be <10MB

# Check what's tracked
git ls-files | wc -l
# Should be ~40-50 files

# Check for large files
git rev-list --objects --all | \
    git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | \
    awk '/^blob/ {print substr($0,6)}' | \
    sort -k2 -n -r | \
    head -10
# Largest file should be <1MB
```

## What Should Be in Git

**Include:**
- Source code (`.py` files)
- Configuration files (`.yaml`, `.json`)
- Documentation (`.md` files)
- Scripts (`.sh` files)
- Requirements (`requirements.txt`)

**Exclude:**
- Models (`*.pth`, `*.safetensors`)
- Data (`data/`, `*.jsonl`)
- Logs (`runs/`, `*.log`)
- Virtual environments (`venv/`, `avs_venv/`)
- Test outputs (`test_*/`, `*_test/`)

## Expected Repository Size

After cleanup:
- `.git` folder: **5-10MB**
- Total repository: **<20MB**
- Push time: **<1 minute**

## Recommended: Option 1 (Start Fresh)

Since you're just starting and haven't pushed successfully yet, **Option 1** is the fastest and cleanest solution:

```bash
# Quick commands
cd ~/Desktop/avs
rm -rf .git
git init
git add .
git commit -m "Initial commit - VLM-MARL system"
git remote add origin <your-repo-url>
git push -u origin main --force
```

This will give you a clean repository with only the essential code files.

## After Cleanup

Once your repository is clean:

```bash
# 1. Verify size
du -sh .git  # Should be <10MB

# 2. Push to remote
git push -u origin main

# 3. Clone on HPC laptop
git clone <your-repo-url> vlm-marl-highway
cd vlm-marl-highway
./setup.sh
```

Your push should complete in <1 minute instead of being stuck at 275MB!
