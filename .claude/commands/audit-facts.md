Audit and fix the fact file at `$ARGUMENTS` against all rules in `private_scripts/harry/facts/README.md`.

This command finds issues, fixes them, prints what changed, and optionally updates the README with new patterns.

---

## Step 1: Automated checks

Read the target fact file and the README. Then run this Python script, replacing `TARGET` with the resolved file path:

```python
import importlib.util, sys, re, os

target = "TARGET"

# Resolve to absolute path
if not os.path.isabs(target):
    target = os.path.join(os.getcwd(), target)

# Load module
spec = importlib.util.spec_from_file_location("facts_module", target)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

P = mod.POSITIVE
L1 = mod.LEVEL_1
L2 = mod.LEVEL_2
L3 = mod.LEVEL_3
CTX = getattr(mod, 'CONTEXT', [])

fails = {}
def fail(check, idx, msg):
    fails.setdefault(check, []).append((idx, msg))

# --- A. Count alignment ---
counts = {'POSITIVE': len(P), 'LEVEL_1': len(L1), 'LEVEL_2': len(L2), 'LEVEL_3': len(L3)}
if len(set(counts.values())) != 1:
    fail('A_COUNT', -1, f'List lengths differ: {counts}')

n = len(P)

# --- B. Duplicate check ---
for name, lst in [('POSITIVE', P), ('LEVEL_1', L1), ('LEVEL_2', L2), ('LEVEL_3', L3)]:
    seen = {}
    for i, s in enumerate(lst):
        if s in seen:
            fail('B_DUPLICATE', i, f'{name}[{i}] duplicates [{seen[s]}]: {repr(s)}')
        seen[s] = i

# --- D. Ambiguous modifiers ---
bad_mods = ['recently', ' still ', 'always', 'at home', 'in interviews']
for i, p in enumerate(P):
    for mod in bad_mods:
        if mod.lower() in p.lower():
            fail('D_AMBIGUOUS_MOD', i, f'Has "{mod.strip()}": {repr(p)}')

# --- E. Leading prepositional phrases (level-specific ambiguity) ---
lead_preps = ['In ', 'At ', 'Across ', 'Since ', 'During ', 'Throughout ', 'According to ']
for i, p in enumerate(P):
    for pp in lead_preps:
        if p.startswith(pp):
            fail('E_LEADING_PREP', i, f'Starts with "{pp.strip()}": {repr(p)}')

# --- F. Embedded meta-phrases ---
meta = ['it is a well known fact', 'it is widely accepted', 'it is clear that',
        'everyone agrees', 'the consensus is', 'it is common knowledge',
        'it is undisputed', 'it is well-known', 'it is widely known']
for i, p in enumerate(P):
    for m in meta:
        if m in p.lower():
            fail('F_META_PHRASE', i, f'Has "{m}": {repr(p)}')

# --- G. Dates in POSITIVE ---
for i, p in enumerate(P):
    if re.search(r'\b\d{2,4}(th|st|nd|rd)?[- ]century\b|\b(1[0-9]{3}|20\d{2})\b', p, re.IGNORECASE):
        fail('G_DATE', i, f'Contains date: {repr(p)}')

# --- H. Terminal period ---
for name, lst in [('POSITIVE', P), ('LEVEL_1', L1), ('LEVEL_2', L2), ('LEVEL_3', L3)]:
    for i, s in enumerate(lst):
        if not s.endswith('.'):
            fail('H_PERIOD', i, f'{name}[{i}] missing terminal period: {repr(s[-30:])}')

# --- I. L2 word-for-word check (sentence negation: "It is false that...") ---
l2_prefixes = ['It is not true that ', 'It is false that ',
               'It is not the case that ', 'It is untrue that ']
for i in range(min(n, len(L2))):
    p = P[i]
    l2 = L2[i]
    p_lower = p[0].lower() + p[1:]
    ok = any(l2 == pfx + p or l2 == pfx + p_lower for pfx in l2_prefixes)
    if not ok:
        fail('I_L2_WRAP', i, f'L2 does not wrap POSITIVE.\n  POS: {repr(p)}\n  L2:  {repr(l2)}')

# --- J. L3 word-for-word check (external negation: "The following is false. ...") ---
for i in range(min(n, len(L3))):
    p = P[i]
    l3 = L3[i]
    if not l3.endswith(p):
        fail('J_L3_WRAP', i, f'L3 does not end with POSITIVE.\n  POS: {repr(p)}\n  L3:  {repr(l3)}')

# --- K. L3 forward reference ---
for i in range(min(n, len(L3))):
    l3 = L3[i]
    prefix = l3.split('. ')[0]
    if 'following' not in prefix.lower() and 'next' not in prefix.lower():
        fail('K_L3_FWD_REF', i, f'L3 prefix lacks "following"/"next": {repr(prefix)}')

# --- L. CONTEXT gender neutrality ---
gender_pats = [r'\bfemale\b', r'\bmale\b', r'\bgirl\b', r'\bboy\b',
               r'\bwoman\b', r'\bman\b', r'\bdaughter\b', r'\bson\b',
               r'\bfeminine\b', r'\bmasculine\b', r'\bshe\b', r'\bhe\b',
               r'\bher\b', r'\bhis\b', r'\bsister\b', r'\bbrother\b']
for i, c in enumerate(CTX):
    for pat in gender_pats:
        if re.search(pat, c, re.IGNORECASE):
            fail('L_CTX_GENDER', i, f'CONTEXT has gender term "{pat}": {repr(c)}')

# --- M. CONTEXT -estre hint ---
for i, c in enumerate(CTX):
    if '-estre' in c.lower() or "'-estre'" in c.lower():
        fail('M_CTX_ESTRE', i, f'CONTEXT references -estre (feminine OE suffix): {repr(c)}')

# --- Print results ---
all_checks = ['A_COUNT', 'B_DUPLICATE', 'D_AMBIGUOUS_MOD',
              'E_LEADING_PREP', 'F_META_PHRASE', 'G_DATE', 'H_PERIOD',
              'I_L2_WRAP', 'J_L3_WRAP', 'K_L3_FWD_REF', 'L_CTX_GENDER', 'M_CTX_ESTRE']

check_names = {
    'A_COUNT': 'Count alignment',
    'B_DUPLICATE': 'Duplicate check',
    'D_AMBIGUOUS_MOD': 'Ambiguous modifiers',
    'E_LEADING_PREP': 'Leading prepositional phrases',
    'F_META_PHRASE': 'Embedded meta-phrases',
    'G_DATE': 'Dates in POSITIVE',
    'H_PERIOD': 'Terminal period',
    'I_L2_WRAP': 'L2 word-for-word wrapping (sentence negation)',
    'J_L3_WRAP': 'L3 word-for-word wrapping (external negation)',
    'K_L3_FWD_REF': 'L3 forward-reference',
    'L_CTX_GENDER': 'CONTEXT gender neutrality',
    'M_CTX_ESTRE': 'CONTEXT -estre hint',
}

print(f'\n=== AUTOMATED AUDIT: {os.path.basename(target)} ===')
print(f'Lists: POSITIVE={len(P)}, LEVEL_1={len(L1)}, LEVEL_2={len(L2)}, LEVEL_3={len(L3)}, CONTEXT={len(CTX)}')
print()

total_fails = 0
for check in all_checks:
    name = check_names[check]
    if check in fails:
        entries = fails[check]
        total_fails += len(entries)
        print(f'FAIL  {name} ({len(entries)} issue{"s" if len(entries) != 1 else ""})')
        for idx, msg in entries:
            print(f'      [{idx}] {msg}')
    else:
        print(f'PASS  {name}')

print(f'\n{"ALL AUTOMATED CHECKS PASSED" if total_fails == 0 else f"TOTAL: {total_fails} issues found"}')
```

---

## Step 2: Semantic audit

After the automated checks, read every POSITIVE/LEVEL_1 entry and check against:

1. **Core claim test** (README §Core Claim Test): Does each L1 negation naturally read as denying the core claim? Watch for:
   - Qualifier-only negation (negating a qualifier but the core claim could still hold) — includes "naturally", "by nature", "practicing", "qualified", "trained", "known as", "by trade", "by profession"
   - Presuppositional negation (achievement verbs that presuppose attempt)
   - Documentation verb negation without "that" clause ("lists X as Y", "identifies X as Y", "classifies X as Y", "describes X as Y", "records X as Y", "includes X as Y") — THE MOST COMMON MISTAKE. These must be restructured with "that" clauses
   - Source verb outer negation on "that" clauses — verify negation is AFTER the word "that", not before it
   - Temporal/developmental framing where negation targets timing, not the claim itself (includes "remains" → "does not remain")
   - Activity ≠ Identity for occupation claims ("runs a practice" → "does not run a practice" doesn't negate being a dentist)
   - L1 word fidelity — verify L1 only inserts negation words, does not change or add other words
   - Multi-site negation — causal chains ("X because Y", "X when Y") that require two negation insertions. Search for "because" and "when" followed by the core claim
   - Pragmatic presupposition of denial — social speech acts ("People say that X is not Y") that imply a prior belief in Y
   - Qualifier scope in POSITIVE — scope qualifiers ("closed to the public") where negation targets the qualifier not the core claim

2. **Structural incompatibility** (README §Structural Incompatibility): Are both entities joined in a prepositional phrase ("between X and Y", "of X and Y") making clean internal negation impossible?

3. **Self-report verb ambiguity** (README §Self-Report Verbs): For verbs without "that" clauses (describes, claims, refers to), is Placement B (inner negation) used? Is there dual-placement ambiguity? Also check for hedging verbs ("claims", "maintains", "insists", "protests").

4. **Level-specific ambiguity** (README §Level-Specific Ambiguity): For entries with modifiers or context phrases, does the L2 sentence negation create ambiguity about what is being negated? Test: wrap with "It is false that [X]" — does this unambiguously negate the core claim? Also check that POSITIVE does not contain negative formulations ("not open", "does not operate") that create double negation in L2.

5. **Source plausibility** (README §Naturalness and plausibility): For each POSITIVE entry that references a source, institution, or context, check:
   - Does this source/context actually exist as a real genre? ("handedness reference guides", "textbooks on handedness", "instructional materials on laterality" = NO)
   - Would this specific entity plausibly appear in this context? (A random fictional person would not be named in scientific literature, peer-reviewed studies, or academic research)
   - Does the source imply the claim is debatable when it should be settled fact? ("Professional evaluations confirm that X is left-handed" implies handedness is contentious; "Experts on handedness say X is left-handed" implies experts are needed for something mundane)
   - For mundane personal traits of ordinary/fictional people (handedness, family structure, birthplace), prefer personal framings (friends, family, colleagues), routine records (medical, personnel, school), and self-report — NOT expert/scientific/research framings

6. **CONTEXT narrative consistency** (README §Narrative consistency): Do any CONTEXT entries contradict POSITIVE facts?

7. **Grammar**: Check all entries across all 4 lists for grammatical errors.

8. **Self-containment** (README §Self-Containment): For fictional entities, does each POSITIVE fact establish what kind of entity is being discussed without relying on context from other facts?

**Output of this step**: A numbered list of every failing entry, quoting the exact text and which rule it violates.

---

## Step 3: Fix all issues

For every issue found in Steps 1 and 2, **fix it directly in the fact file**. Maintain list alignment — all 4 lists must stay index-aligned.

### Fixing strategies (in order of preference)

1. **Repair in place** (best when possible): If the POSITIVE is fine but L1/L2/L3 is wrong, fix the derived entry:
   - L2/L3 wrapping errors → regenerate mechanically from POSITIVE
   - L1 outer negation on "that" clause → move negation inside the "that" clause
   - L1 word fidelity (changed/added words) → regenerate L1 from POSITIVE, inserting only negation
   - Missing terminal period → add it

2. **Replace the POSITIVE and regenerate L1/L2/L3**: If the POSITIVE itself is the problem (qualifier, causal chain, documentation verb, implausible source, etc.):
   - Write a new POSITIVE that follows all README rules
   - Generate L1: insert negation between the two core entities
   - Generate L2: mechanically apply the next prefix in the cycling sequence
   - Generate L3: mechanically apply the next prefix in the cycling sequence
   - The replacement should fit the same diversity category as the entry it replaces
   - Avoid duplicating existing entries

3. **Remove the entry from all 4 lists**: Only as a last resort if no valid replacement can be found. This reduces the total count, which is acceptable but not ideal.

### Rules for fixing

- **Every fix must touch all 4 lists at the same index.** If you change POSITIVE[i], you must also update LEVEL_1[i], LEVEL_2[i], and LEVEL_3[i].
- **L2 is always mechanical**: `prefix + POSITIVE[i]` (lowercase first letter unless proper noun). Cycle through the 4 prefixes.
- **L3 is always mechanical**: `prefix + " " + POSITIVE[i]` (preserve original casing). Cycle through the 15 prefixes.
- **L1 must pass the core claim test**: the negation must naturally read as denying the core claim.
- **Do not introduce new issues** while fixing. Each replacement must satisfy all rules in the README.

### Print each fix

For every change, print in this format:

```
FIX [index]: <rule violated>
  OLD POSITIVE: "..."
  NEW POSITIVE: "..."
  OLD L1: "..."
  NEW L1: "..."
  (L2/L3 regenerated mechanically)
```

For L1-only repairs (e.g., moving negation inside a "that" clause):

```
FIX [index]: Source verb outer negation
  POSITIVE: "..." (unchanged)
  OLD L1: "..."
  NEW L1: "..."
```

---

## Step 4: Re-run automated checks

After making all fixes, re-run the full automated audit script from Step 1 on the modified file. If any checks fail, go back to Step 3 and fix them. Repeat until all automated checks pass.

---

## Step 5: Update the README (if warranted)

Review all the issues found and fixed. If you encountered a **new failure pattern** that is not already documented in `private_scripts/harry/facts/README.md`, add it:

- Add a short subsection describing the pattern, with an example from the file you just fixed
- Add a row to the **Audit Findings Summary** table at the bottom of the README
- Only add patterns that are likely to recur in future fact files — don't document one-off typos or grammar errors

**When NOT to update the README**:
- The issue is already documented (most will be)
- The issue was a one-off mistake (typo, grammar) rather than a systematic pattern
- The issue is too specific to this one fact file to generalize

If you do update the README, print what you added:

```
README UPDATE: Added §<section name>
  Pattern: <brief description>
  Example: "..."
```

If no README updates are warranted, print:

```
README: No new patterns found — all issues matched existing documentation.
```

---

## Step 6: Summary

Print a summary:

```
=== AUDIT COMPLETE: {filename} ===
Issues found: N (M automated, K semantic)
Issues fixed: N
  - Replaced: X entries (new POSITIVE + regenerated L1/L2/L3)
  - Repaired: Y entries (L1/L2/L3 fixed, POSITIVE unchanged)
  - Removed: Z entries (from all 4 lists)
Final counts: POSITIVE=N, LEVEL_1=N, LEVEL_2=N, LEVEL_3=N
README updates: N new patterns added
All automated checks: PASS
```
