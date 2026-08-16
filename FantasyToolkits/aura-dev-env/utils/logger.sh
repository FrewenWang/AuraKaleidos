#!/bin/bash
# ============================================================
# Logging Functions
# ============================================================

# Source colors
source "$(dirname "$0")/colors.sh"

log_error() {
  echo -e "${RED}❌ Error:${NC} $*" >&2
}

log_warn() {
  echo -e "${YELLOW}⚠️  Warning:${NC} $*" >&2
}

log_info() {
  echo -e "${BLUE}ℹ️  Info:${NC} $*" >&2
}

log_success() {
  echo -e "${GREEN}✅ Success:${NC} $*"
}
