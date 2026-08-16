#!/bin/bash
# ============================================================
# AURA_ENV Shell Initialization Script
# Safe initialization with error handling and recovery
# ============================================================

# 注意：此处故意不使用 set -e
# 被加载的 .fwf 配置使用 `[ -d "..." ] && ...` 模式，目录不存在时返回 1
# 若启用 set -e，会导致整个初始化链意外退出（违背"缺失目录不破坏系统"原则）

# ============================================================
# Setup
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../utils/logger.sh
source "$SCRIPT_DIR/../utils/logger.sh"

# ============================================================
# Configuration
# ============================================================

AURA_ENV_ROOT="${1:-.}"
REQUIRED_FILES=(
  "$AURA_ENV_ROOT/aura-dev-env.sh"
)

# Optional files are loaded directly in initialize_aura_env
OPTIONAL_FILES=()

# ============================================================
# Helper Functions
# ============================================================

# Safe source with error handling
safe_source() {
  local file="$1"
  local optional="${2:-false}"

  if [ ! -f "$file" ]; then
    if [ "$optional" = "true" ]; then
      log_warn "Optional file not found: $file"
      return 0 # Don't fail for optional files
    else
      log_error "Required file not found: $file"
      return 1
    fi
  fi

  if source "$file" 2> /dev/null; then
    log_info "Loaded: $file"
    return 0
  else
    if [ "$optional" = "true" ]; then
      log_warn "Failed to load optional file: $file"
      return 0
    else
      log_error "Failed to load required file: $file"
      return 1
    fi
  fi
}

# ============================================================
# Validation
# ============================================================

validate_environment() {
  log_info "Validating AURA_ENV installation..."

  # Check if root directory exists
  if [ ! -d "$AURA_ENV_ROOT" ]; then
    log_error "AURA_ENV_ROOT not found: $AURA_ENV_ROOT"
    return 1
  fi

  # Check required files
  local all_required_ok=true
  for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
      log_error "Missing required file: $file"
      all_required_ok=false
    fi
  done

  if [ "$all_required_ok" != "true" ]; then
    return 1
  fi

  log_success "Validation passed"
  return 0
}

# ============================================================
# Initialization
# ============================================================

initialize_aura_env() {
  log_info "Initializing AURA_ENV..."

  # Load required files
  for file in "${REQUIRED_FILES[@]}"; do
    if ! safe_source "$file" "false"; then
      return 1
    fi
  done

  # Load optional files (don't fail if missing)
  for file in "${OPTIONAL_FILES[@]}"; do
    safe_source "$file" "true"
  done

  # Load platform-specific config based on OS
  local platform_config="$AURA_ENV_ROOT/bash/base/bashrc_$(uname -s).fwf"
  if [ -f "$platform_config" ]; then
    safe_source "$platform_config" "true"
  fi

  log_success "AURA_ENV initialized successfully"
  return 0
}

# ============================================================
# Recovery / Repair
# ============================================================

repair_installation() {
  log_warn "Attempting to repair AURA_ENV installation..."

  # Check if setup.sh exists
  if [ -f "$AURA_ENV_ROOT/setup.sh" ]; then
    log_info "Running setup.sh..."
    if bash "$AURA_ENV_ROOT/setup.sh" > /dev/null 2>&1; then
      log_success "Repair completed"
      return 0
    else
      log_error "Repair failed"
      return 1
    fi
  else
    log_error "Cannot repair: setup.sh not found"
    return 1
  fi
}

# ============================================================
# Main
# ============================================================

main() {
  export AURA_ENV_ROOT

  # Try to validate
  if ! validate_environment; then
    log_error "Installation validation failed"

    # Try to repair
    if repair_installation; then
      log_info "Retrying initialization after repair..."
      if ! validate_environment; then
        log_error "Validation still fails after repair"
        return 1
      fi
    else
      log_error "Cannot repair, giving up"
      return 1
    fi
  fi

  # Initialize
  if initialize_aura_env; then
    return 0
  else
    log_error "Initialization failed"
    return 1
  fi
}

main "$@"
