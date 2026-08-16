#!/bin/bash
# ============================================================
# AURA_ENV Shell Initialization Repair Tool
# Diagnoses and repairs shell initialization issues
# ============================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

AURA_ENV_ROOT="${1:-.}"

log_info "Checking AURA_ENV shell initialization..."
log_info "AURA_ENV_ROOT: $AURA_ENV_ROOT"
echo ""

# Check if root directory exists
if [ ! -d "$AURA_ENV_ROOT" ]; then
  log_error "AURA_ENV_ROOT not found: $AURA_ENV_ROOT"
  echo ""
  echo "Please set correct AURA_ENV_ROOT:"
  echo "  bash $0 /path/to/aura-dev-env"
  exit 1
fi

# Check required files
MISSING_FILES=()
BROKEN_FILES=()

log_info "Checking required files..."

if [ ! -f "$AURA_ENV_ROOT/aura-dev-env.sh" ]; then
  MISSING_FILES+=("$AURA_ENV_ROOT/aura-dev-env.sh")
  log_error "Missing: aura-dev-env.sh"
else
  if bash -n "$AURA_ENV_ROOT/aura-dev-env.sh" 2> /dev/null; then
    log_success "Present: aura-dev-env.sh"
  else
    BROKEN_FILES+=("$AURA_ENV_ROOT/aura-dev-env.sh")
    log_error "Broken: aura-dev-env.sh (syntax error)"
  fi
fi

echo ""
log_info "Checking platform-specific config..."

# Check platform-specific config (dynamic based on current OS)
platform=$(uname -s)
platform_config="$AURA_ENV_ROOT/bash/base/bashrc_${platform}.fwf"

if [ ! -f "$platform_config" ]; then
  log_warn "Missing: bash/base/bashrc_${platform}.fwf"
else
  log_success "Present: bash/base/bashrc_${platform}.fwf"
fi

echo ""

# Summary
if [ -z "${MISSING_FILES[*]}" ] && [ -z "${BROKEN_FILES[*]}" ]; then
  log_success "All required files present and valid"
  echo ""
  echo "Your shell initialization is OK!"
  echo "Try running: source ~/.bashrc"
  exit 0
fi

# If there are issues, offer to repair
if [ -n "${MISSING_FILES[*]}" ] || [ -n "${BROKEN_FILES[*]}" ]; then
  log_error "Issues found:"
  for file in "${MISSING_FILES[@]}"; do
    echo "  - Missing: $file"
  done
  for file in "${BROKEN_FILES[@]}"; do
    echo "  - Syntax Error: $file"
  done
  echo ""

  # Try to repair
  if [ -f "$AURA_ENV_ROOT/setup.sh" ]; then
    log_warn "Attempting repair with setup.sh..."
    echo ""

    if bash "$AURA_ENV_ROOT/setup.sh"; then
      echo ""
      log_success "Repair completed successfully!"
      echo ""
      echo "Next steps:"
      echo "  1. source ~/.bashrc"
      echo "  2. aura-env validate"
    else
      log_error "Setup failed"
      exit 1
    fi
  else
    log_error "Cannot repair: setup.sh not found at $AURA_ENV_ROOT/setup.sh"
    exit 1
  fi
fi
