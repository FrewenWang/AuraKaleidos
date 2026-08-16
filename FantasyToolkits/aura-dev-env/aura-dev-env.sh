#!/usr/bin/env sh
# AURA_ENV environment setup (must be sourced)

# ===============================
# 1. Ensure script is sourced
# ===============================

is_sourced=1

if [ -n "$BASH_SOURCE" ]; then
  if [ "$BASH_SOURCE" != "$0" ]; then
    is_sourced=0
  fi
elif [ -n "$ZSH_EVAL_CONTEXT" ]; then
  case $ZSH_EVAL_CONTEXT in
    *:file) is_sourced=0 ;;
    *)      is_sourced=1 ;;
  esac
else
  is_sourced=1
fi

if [ "$is_sourced" -eq 1 ]; then
  echo "ERROR: AURA_ENV-env.sh must be sourced:"
  echo "  source AURA_ENV-env.sh"
  return 1 2>/dev/null || exit 1
fi

# ===============================
# 2. Resolve script path
# ===============================

if [ -n "$BASH_SOURCE" ]; then
  SCRIPT_SOURCE="$BASH_SOURCE"
elif [ -n "$ZSH_VERSION" ]; then
  SCRIPT_SOURCE="${(%):-%x}"
else
  echo "Unsupported shell (bash/zsh required)"
  return 1
fi

AURA_ENV_BASE="$(cd "$(dirname "$SCRIPT_SOURCE")" && pwd)"
export AURA_ENV_ROOT="$AURA_ENV_BASE"

# ===============================
# 3. Platform-specific paths
# ===============================

HOST_OS="$(uname -s)"

AURA_ENV_HOST="$AURA_ENV_BASE/bin/main"

case "$HOST_OS" in
  Linux)
    AURA_ENV_HOST_TARGET="$AURA_ENV_BASE/bin/main/Linux"
    ;;
  Darwin)
    AURA_ENV_HOST_TARGET="$AURA_ENV_BASE/bin/main/Darwin"
    ;;
  *)
    AURA_ENV_HOST_TARGET="$AURA_ENV_BASE"
    ;;
esac

# ===============================
# 4. Load environment-specific bash config
# ===============================

# Load base bash configuration (always)
if [ -f "$AURA_ENV_BASE/bash/base/bashrc.fwf" ]; then
  # shellcheck source=/dev/null
  source "$AURA_ENV_BASE/bash/base/bashrc.fwf"
fi

# Load platform-specific bash configuration (if available)
PLATFORM="$(uname -s)"
if [ -f "$AURA_ENV_BASE/bash/base/bashrc_${PLATFORM}.fwf" ]; then
  # shellcheck source=/dev/null
  source "$AURA_ENV_BASE/bash/base/bashrc_${PLATFORM}.fwf"
fi

# Load environment-specific bash configuration
CONFIG_FILE="$HOME/.aura-env-config"
SELECTED_ENV="base"
if [ -f "$CONFIG_FILE" ]; then
  # shellcheck source=/dev/null
  source "$CONFIG_FILE"
  SELECTED_ENV="${AURA_ENV:-base}"
fi

if [ "$SELECTED_ENV" != "base" ] && [ -f "$AURA_ENV_BASE/bash/${SELECTED_ENV}/bashrc.fwf" ]; then
  # shellcheck source=/dev/null
  source "$AURA_ENV_BASE/bash/${SELECTED_ENV}/bashrc.fwf"
fi

# Load environment-specific platform bash configuration (if available)
if [ "$SELECTED_ENV" != "base" ] && [ -f "$AURA_ENV_BASE/bash/${SELECTED_ENV}/bashrc_${PLATFORM}.fwf" ]; then
  # shellcheck source=/dev/null
  source "$AURA_ENV_BASE/bash/${SELECTED_ENV}/bashrc_${PLATFORM}.fwf"
fi

# ===============================
# 5. Build environment-specific bin PATH
# ===============================

build_bin_path() {
  local env_name="$1"
  local aura_root="$2"
  local existing_path="$3"
  local platform="$(uname -s)"

  # Build path with 4-layer precedence (only add directories that exist):
  # 1. Environment + Platform specific (if exists)
  # 2. Environment specific (if exists)
  # 3. Base + Platform specific (if exists)
  # 4. Base (if exists)
  # 5. Main binaries (if exists)
  # 6. Existing PATH

  local new_path=""
  local env_platform_bin="$aura_root/bin/$env_name/$platform"
  local env_bin="$aura_root/bin/$env_name"
  local base_platform_bin="$aura_root/bin/base/$platform"
  local base_bin="$aura_root/bin/base"
  local main_bin="$aura_root/bin/main"

  # Only add directories that exist
  [ -d "$env_platform_bin" ] && new_path="$new_path:$env_platform_bin"
  [ -d "$env_bin" ] && new_path="$new_path:$env_bin"
  [ -d "$base_platform_bin" ] && new_path="$new_path:$base_platform_bin"
  [ -d "$base_bin" ] && new_path="$new_path:$base_bin"
  [ -d "$main_bin" ] && new_path="$new_path:$main_bin"
  [ -n "$existing_path" ] && new_path="$new_path:$existing_path"

  # Remove leading colon and duplicates while preserving order
  new_path="${new_path#:}"
  echo "$new_path" | tr ':' '\n' | awk '!seen[$0]++' | tr '\n' ':' | sed 's/:$//'
}

# Apply environment-specific bin paths
# Always process regardless of whether directories exist - build_bin_path handles missing dirs
SELECTED_ENV="${AURA_ENV:-base}"
if [ "$SELECTED_ENV" = "default" ] || [ "$SELECTED_ENV" = "base" ]; then
  SELECTED_ENV="base"
fi

PATH=$(build_bin_path "$SELECTED_ENV" "$AURA_ENV_BASE" "$AURA_ENV_HOST:$AURA_ENV_HOST_TARGET:$PATH")

# ===============================
# 6. Export environment (idempotent)
# ===============================

case ":$PATH:" in
  *":$AURA_ENV_HOST:"*) ;;
  *) PATH="$AURA_ENV_HOST:$PATH" ;;
esac

case ":$PATH:" in
  *":$AURA_ENV_HOST_TARGET:"*) ;;
  *) PATH="$AURA_ENV_HOST_TARGET:$PATH" ;;
esac

export AURA_ENV_BASE AURA_ENV_HOST AURA_ENV_HOST_TARGET PATH

# ===============================
# 6. Ensure executables
# ===============================

[ -d "$AURA_ENV_HOST" ] && chmod -R a+x "$AURA_ENV_HOST"/* 2>/dev/null || true
[ -d "$AURA_ENV_HOST_TARGET" ] && chmod -R a+x "$AURA_ENV_HOST_TARGET"/* 2>/dev/null || true

# ===============================
# 7. Create aura-env command alias
# ===============================

# Create a temporary wrapper script in /tmp to avoid zsh printing function definitions
# This file will be cleaned up when the shell exits
AURA_ENV_WRAPPER="/tmp/aura-env-wrapper-$$"
cat > "$AURA_ENV_WRAPPER" << 'WRAPPER_EOF'
#!/bin/bash
exec bash "$AURA_ENV_ROOT/scripts/env-manager.sh" "$@"
WRAPPER_EOF
chmod +x "$AURA_ENV_WRAPPER"

# Clean up the wrapper file when the shell exits
trap 'rm -f "$AURA_ENV_WRAPPER"' EXIT 2>/dev/null || true

# Add wrapper to PATH so it can be called as a command
# This avoids the need for a function definition
case ":$PATH:" in
  *":$AURA_ENV_WRAPPER:"*) ;;
  *) PATH="$AURA_ENV_WRAPPER:$PATH" ;;
esac

# Define aura-env function silently
# Use exec to suppress function definition output in zsh
exec 3>&1 4>&2 >/dev/null 2>&1
aura-env() {
  "$AURA_ENV_WRAPPER" "$@"
}
export -f aura-env
exec 1>&3 2>&4 3>&- 4>&-
