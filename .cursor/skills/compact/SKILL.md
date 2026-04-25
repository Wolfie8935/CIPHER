---
name: compact
description: Compacts long conversations into a concise handoff summary for the next run. Use when the user asks to compact, summarize for continuation, reduce token usage, or prepare context for a follow-up session.
---

# Compact

Use this skill to produce a low-token, high-signal conversation handoff.

## When to Use

- User asks for `/compact`, "compact this chat", "summarize for next run", or "reduce tokens"
- Conversation is long and implementation context must be preserved
- A continuation handoff is needed for another agent/session

## Output Rules

- Keep output concise and factual
- Preserve decisions, constraints, and unresolved items
- Prefer bullets over prose
- Avoid repeating full logs, stack traces, or unchanged context
- Include exact file paths and symbols when relevant

## Required Structure

Produce these sections in order:

1. `Primary Goal`
2. `What Was Done`
3. `Key Decisions`
4. `Files Changed`
5. `Current State`
6. `Open Issues / Risks`
7. `Next Actions`

## Compression Checklist

- Include only information needed to continue work correctly
- Keep implementation details only where behavior depends on them
- Collapse repetitive attempts into one final outcome
- Mark assumptions explicitly
- Separate completed work from pending work

## Formatting Template

```markdown
Primary Goal
- ...

What Was Done
- ...

Key Decisions
- ...

Files Changed
- `path/to/file`: brief purpose

Current State
- ...

Open Issues / Risks
- ...

Next Actions
- ...
```

## Quality Bar

Before finalizing, verify:

- A new agent could continue immediately using only this summary
- No critical constraint is missing
- No irrelevant detail inflates token count
