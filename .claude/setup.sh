#!/bin/bash
# Claude Code memory setup — run once per new workspace
# Links repo memory files so Claude Code can read/write them

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MEMORY_SRC="${REPO_DIR}/.claude/memory"
MEMORY_DST="/root/.claude/projects/-source-inyup/memory"

mkdir -p "$(dirname "${MEMORY_DST}")"
ln -sf "${MEMORY_SRC}" "${MEMORY_DST}"

echo "Claude Code memory linked: ${MEMORY_SRC} → ${MEMORY_DST}"
