---
name: Never overwrite sihun data
description: NEVER write/overwrite files under /data/sihun/ or /source/sihun/ paths — always create separate output paths
type: feedback
---

Never overwrite files in /data/sihun/ or /source/sihun/ paths. These are shared data owned by the advisor (sihun/sihun cha).

**Why:** Accidentally overwrote operators.pkl files in /data/sihun/multiface_align/precomputes/ with cupy-incompatible versions. User was very upset.

**How to apply:** Always create a separate output directory (e.g. /data/inyup/, or a new subfolder) when generating/saving any files. Never use sihun paths as output destinations. Ask user before writing to any shared path.
