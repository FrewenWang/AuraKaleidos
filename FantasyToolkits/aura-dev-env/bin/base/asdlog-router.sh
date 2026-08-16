#!/bin/bash
# ASDLOG environment-aware router
#
# This script automatically selects the correct asdlog version
# based on the current development environment.
#
# Versions:
#   asdlog.base     - Base implementation (default)
#   asdlog.dev      - Development (with verbose logging)
#   asdlog.learn    - Learning (simplified)
#   asdlog.corporate - Corporate (with compliance checks)
#
# Usage: asdlog [options]
# The router will automatically pass all arguments to the correct version.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_NAME="asdlog"

# Get current environment
get_environment() {
  local config_file="$HOME/.aura-env-config"
  if [ -f "$config_file" ]; then
    grep "^AURA_ENV=" "$config_file" 2> /dev/null | cut -d'=' -f2 | tr -d '"' || echo "base"
  else
    echo "base"
  fi
}

# Route to the correct version
route_asdlog() {
  local current_env=$(get_environment)
  local env_specific="$SCRIPT_DIR/$SCRIPT_NAME.$current_env"
  local base_version="$SCRIPT_DIR/$SCRIPT_NAME.base"

  # Debug: Show which version is being used (optional)
  if [ "$ASDLOG_DEBUG" = "1" ]; then
    echo "ASDLOG Router: Using environment '$current_env'" >&2
  fi

  # Try environment-specific version
  if [ -f "$env_specific" ] && [ -x "$env_specific" ]; then
    if [ "$ASDLOG_DEBUG" = "1" ]; then
      echo "ASDLOG Router: Using $env_specific" >&2
    fi
    exec "$env_specific" "$@"
  fi

  # Fall back to base version
  if [ -f "$base_version" ] && [ -x "$base_version" ]; then
    if [ "$ASDLOG_DEBUG" = "1" ]; then
      echo "ASDLOG Router: Using $base_version" >&2
    fi
    exec "$base_version" "$@"
  fi

  # Error if nothing found
  echo "Error: ASDLOG implementation not found" >&2
  echo "Looking for: $env_specific or $base_version" >&2
  echo "Current environment: $current_env" >&2
  exit 1
}

# Execute router
route_asdlog "$@"
