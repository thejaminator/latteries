# Git Workflow Reminder for Claude

**IMPORTANT: When the user asks to push to main and sync to public, follow these steps:**

## Step 1: Check Changes
```bash
git status
git diff --stat
git log -5 --oneline
```

## Step 2: Stage and Commit
```bash
git add -A
git commit -m "message"
```

## Step 3: Push to Private Main
```bash
git push origin main
```

## Step 4: Sync to Public Main
**DO NOT** use `git push public main` directly!

Instead, use the sync script:
```bash
echo "y" | .git/sync-to-public.sh
```

This script:
- Automatically excludes `private_scripts/` and `results_dump/`
- Creates a temp branch from public/main
- Copies all non-private files from main
- Creates a sync commit and pushes to public

## Key Files to Reference
- `.claude/git_push.md` - Detailed explanation of the git workflow
- `.git/sync-to-public.sh` - The sync script itself
- `.git/hooks/pre-push` - Protection against accidentally pushing private files

## Common Mistake to Avoid
❌ Don't run: `git push public main` (will fail or push private files)
✅ Do run: `echo "y" | .git/sync-to-public.sh`

## Why This Workflow?
- This is a monorepo with two remotes: private (origin) and public
- Private files must never be pushed to public
- The sync script ensures safe syncing by copying only public files
- Pre-push hooks provide additional protection
