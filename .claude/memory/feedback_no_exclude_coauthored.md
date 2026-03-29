---
name: feedback_no_exclude_coauthored
description: When user says "don't commit that" about untracked files, they mean exclude those files — not remove Co-Authored-By
type: feedback
---

When the user says "그건 커밋하지마" about an untracked file/directory listed in git status, they mean to exclude that path from the commit — NOT to remove the Co-Authored-By line.

**Why:** Misinterpreted "don't commit that" as referring to the co-author tag instead of the untracked directory.
**How to apply:** When ambiguous, interpret "don't commit X" as referring to specific files/directories mentioned in the conversation, not commit message metadata.
