# Contributing Guide

## Repository Structure

This repository has two remotes:
- **origin**: Private mono repository (includes all files including private_scripts/)
- **public**: Public repository (excludes private_scripts/ and results_dump/)

## Workflow for Editing Private Scripts

### Simple Workflow

**If you're ONLY editing private_scripts/ or results_dump/:**

```bash
# 1. Make your changes
vim private_scripts/some_file.py

# 2. Commit
git commit -am "Update private analysis"

# 3. Push to private remote only
git push origin main
# or use the alias:
git push-private
```

### Mixed Changes Workflow

**If you're editing BOTH public and private files:**

**Option 1: Separate commits (Recommended)**
```bash
# Commit public changes
git add example_scripts/ README.md
git commit -m "Update public examples"

# Commit private changes separately
git add private_scripts/
git commit -m "Update private analysis"

# Push both commits to origin
git push origin main

# Push only the public commit to public remote
git push-public-only
# This will interactively show you which commits are safe and let you push them
```

**Option 2: Push everything to origin, then sync public manually**
```bash
# Commit everything together
git commit -am "Update analysis and examples"

# Push to origin (includes everything)
git push origin main

# Later, when you have public-only commits, sync them:
git push-public-only
```

## Git Aliases

The repository is configured with these helpful aliases:

- `git push-private` - Pushes to origin (private remote)
- `git push-public-only` - Intelligently pushes only commits without private files to public

## Protection Mechanism

A pre-push hook automatically prevents pushing private files to the public remote:
- Located at: `.git/hooks/pre-push`
- Blocks any push to `public` remote that contains `private_scripts/` or `results_dump/`
- Provides helpful error messages with suggested alternatives

## What Gets Blocked?

Files/directories that should NEVER go to public:
- `private_scripts/` - Private analysis and research scripts
- `results_dump/` - Experiment results and data dumps

## Troubleshooting

### "ERROR: Attempting to push private files to public remote!"

This means your recent commits include private files. Options:

1. **Push to private only**: `git push origin main`
2. **Use smart push**: `git push-public-only` (pushes only safe commits)
3. **Split your commits**: Separate public and private changes into different commits

### Setting up on a fresh clone

If you clone this repository fresh, you'll need to:

1. Set up the pre-push hook (copy from `.git/hooks/pre-push.sample` or recreate)
2. Configure the git aliases:
   ```bash
   git config alias.push-public-only '!.git/push-public-helper.sh'
   git config alias.push-private 'push origin main'
   ```

## Best Practices

1. ✓ Always push to `origin` first before `public`
2. ✓ Keep public and private changes in separate commits when possible
3. ✓ Use `git push-private` for private-only changes
4. ✓ Use `git push-public-only` when you have mixed commits
5. ✓ Never force push to `public` without verifying no private files are included
