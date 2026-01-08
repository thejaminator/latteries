# Git Push Policy

## ⚠️ CRITICAL RULE: DO NOT PUSH UNLESS USER EXPLICITLY SAYS SO

**Default behavior:** After committing changes, WAIT for the user to tell you to push.

## What This Means

### ❌ DON'T Do This
```bash
# After making changes and committing...
git add -A
git commit -m "message"
git push origin main && echo "y" | .git/sync-to-public.sh  # ❌ NO! Don't auto-push!
```

### ✅ DO This Instead
```bash
# After making changes and committing...
git add -A
git commit -m "message"
# STOP HERE - tell the user what you committed and wait
```

Then tell the user:
> "Committed changes. Let me know if you want me to push to the remotes."

## When to Push

ONLY push when the user explicitly says:
- "push"
- "push to main"
- "push to both remotes"
- "sync to public"
- "pls push"
- Any other clear instruction to push

## Why This Policy?

- User may want to review commits before pushing
- User may want to make additional changes
- User may want to squash commits
- User may want to test locally first
- Pushing is irreversible (especially to public repo)

## The Workflow

1. **Make changes** (edits, new files, etc.)
2. **Stage and commit** with clear commit message
3. **STOP and report** what was committed
4. **WAIT** for user instruction
5. **Push only if instructed** by the user

## Exception

The ONLY exception is if the user explicitly says "and push" or similar in their original request:
- "make this change and push to main" ✓ OK to push
- "update the README" ✗ Don't push, just commit

## Example Interaction

**User:** "Update the README to fix the typo"

**Claude:**
- Fixes the typo
- Commits the change
- Says: "Fixed the typo in README. Committed the change. Let me know if you want me to push."
- WAITS for user instruction

**User:** "ok push"

**Claude:**
- Runs push commands

## Summary

**Commit freely. Push only when told.**
