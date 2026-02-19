# Kill Process

Safely kill specific training processes without collateral damage.

## Rules

**NEVER** use broad patterns like `grep <keyword> | xargs kill` or `pkill -f <pattern>`.

**ALWAYS** follow this procedure:

1. **List first**: Run `ps aux | grep <pattern> | grep -v grep` and display the full output
2. **Review**: Identify which PIDs you actually want to kill vs which should keep running
3. **Confirm with user** if there are more processes matching than expected
4. **Kill by exact PID**: `kill <pid1> <pid2>` - only the specific PIDs confirmed

## Example of what NOT to do

```bash
# BAD - kills everything matching the pattern
ps aux | grep narrow_conscious | grep -v grep | awk '{print $2}' | xargs kill
```

## Example of correct approach

```bash
# GOOD - list first
ps aux | grep narrow_conscious | grep -v grep

# Review output, identify PIDs, then kill only the ones you want
kill 12345 12346  # Only the specific PIDs
```

## Why this matters

Training jobs can run for hours. Killing the wrong process means losing all that progress. A DeepSeek Jesus control at 41% completion was accidentally killed this way.
