#!/bin/bash
# AURA_ENV SDK installer - Compatible with bash and zsh
# Usage: bash setup.sh  (works in any shell environment)

set -e

# ===============================
# 1. Detect OS & user shell
# ===============================

HOST_OS="$(uname -s)"
USER_SHELL="${SHELL:-/bin/bash}"

echo "Host OS    : $HOST_OS"
echo "User shell : $USER_SHELL"

# Determine shell type and config file
SHELL_TYPE=""
BASH_FILE=""

case "$USER_SHELL" in
  */bash | *bash)
    SHELL_TYPE="bash"
    case "$HOST_OS" in
      Linux) BASH_FILE="$HOME/.bashrc" ;;
      Darwin) BASH_FILE="$HOME/.bash_profile" ;;
      *)
        echo "Unsupported OS"
        exit 1
        ;;
    esac
    ;;
  */zsh | *zsh)
    SHELL_TYPE="zsh"
    BASH_FILE="$HOME/.zshrc"
    ;;
  *)
    echo "Unable to determine shell from: $USER_SHELL"
    echo "Trying bash as fallback..."
    SHELL_TYPE="bash"
    case "$HOST_OS" in
      Linux) BASH_FILE="$HOME/.bashrc" ;;
      Darwin) BASH_FILE="$HOME/.bash_profile" ;;
      *)
        echo "Unsupported OS"
        exit 1
        ;;
    esac
    ;;
esac

echo "Shell type : $SHELL_TYPE"
echo "Config file: $BASH_FILE"

# ===============================
# 2. Resolve AURA_ENV location
# ===============================

ENV_FILE="aura-dev-env.sh"
AURA_ENV_ROOT="$(cd "$(dirname "$0")" && pwd)"
AURA_ENV="$AURA_ENV_ROOT/$ENV_FILE"

if [ ! -f "$AURA_ENV" ]; then
  echo "ERROR: $ENV_FILE not found in $AURA_ENV_ROOT"
  exit 1
fi

# ===============================
# 3. Detect multi-environment support (from bash/ directory)
# ===============================

get_environments() {
  local config_file="$AURA_ENV_ROOT/config/environments.conf"

  if [ -f "$config_file" ]; then
    grep -v "^#" "$config_file" | grep -v "^$" | tr '\n' ' ' | sed 's/ $//'
    return
  fi

  # 降级到自动检测
  local bash_dir="$AURA_ENV_ROOT/bash"
  local envs=""

  if [ -d "$bash_dir" ]; then
    for dir in "$bash_dir"/*; do
      if [ -d "$dir" ]; then
        local name=$(basename "$dir")
        if [ "$name" != "base" ] && [ "$name" != "Linux" ] && [ "$name" != "Darwin" ]; then
          envs="$envs $name"
        fi
      fi
    done
  fi

  echo "$envs"
}

if [ -d "$AURA_ENV_ROOT/bash" ]; then
  echo "Multi-environment support detected!"
  echo ""
  echo "Available environments:"
  for env in $(get_environments); do
    echo "  - $env"
  done

  DEFAULT_ENV="${1:-dev}"
  echo ""
  echo "Select environment (default: $DEFAULT_ENV):"
  echo "  base - Base configuration only"
  echo "  dev  - Development environment"
  echo "  mi   - Mobile Infrastructure environment"
  echo ""
fi

# ===============================
# 4. Write rc file (idempotent)
# ===============================

MARK_BEGIN="# >>> AURA_ENV SDK >>>"
MARK_END="# <<< AURA_ENV SDK <<<"

ENV_BLOCK=$(
  cat << 'ENVEOF'
# ============================================================
# AURA_ENV SDK initialization (Safe)
# ============================================================

export AURA_ENV_ROOT='AURA_ENV_ROOT_VALUE'

# Initialize AURA_ENV (safe, never blocks the shell)
# 设计理念：缺失的目录/工具不应破坏系统，配置应优雅降级
if [ -f "$AURA_ENV_ROOT/aura-dev-env.sh" ]; then
    source "$AURA_ENV_ROOT/aura-dev-env.sh" || true
fi
ENVEOF
)

# Replace placeholder with actual path (POSIX compatible)
ENV_BLOCK=$(printf '%s\n' "$ENV_BLOCK" | sed "s|AURA_ENV_ROOT_VALUE|$AURA_ENV_ROOT|g")

if [ -f "$BASH_FILE" ] && grep -q "$MARK_BEGIN" "$BASH_FILE"; then
  echo "✅ AURA_ENV SDK already installed in $BASH_FILE"
else
  echo "📝 Installing AURA_ENV SDK environment to $BASH_FILE..."

  # Create backup if file exists
  if [ -f "$BASH_FILE" ]; then
    cp "$BASH_FILE" "$BASH_FILE.backup.$(date +%s)"
    echo "   ℹ️  Backed up to $BASH_FILE.backup.*"
  fi

  # Write configuration
  if {
    echo ""
    echo "$MARK_BEGIN"
    echo "$ENV_BLOCK"
    echo "$MARK_END"
  } >> "$BASH_FILE" 2> /dev/null; then
    echo "   ✅ Configuration written successfully"
  else
    echo "   ❌ Failed to write to $BASH_FILE"
    echo "   🔧 Trying alternative method..."

    # Try using tee as alternative
    {
      echo ""
      echo "$MARK_BEGIN"
      echo "$ENV_BLOCK"
      echo "$MARK_END"
    } | tee -a "$BASH_FILE" > /dev/null 2>&1

    if grep -q "$MARK_BEGIN" "$BASH_FILE"; then
      echo "   ✅ Configuration written (alternative method)"
    else
      echo "   ❌ ERROR: Could not write to $BASH_FILE"
      echo "   💡 Suggestions:"
      echo "      • Check file permissions: ls -la $BASH_FILE"
      echo "      • Try running: chmod u+w $BASH_FILE"
      echo "      • Or manually copy content from docs/快速开始.md"
      exit 1
    fi
  fi
fi

# ===============================
# 5. Set default environment (optional)
# ===============================

if [ -f "$AURA_ENV_ROOT/config/environments.conf" ] && [ -n "$DEFAULT_ENV" ]; then
  echo ""
  echo "Setting default environment to: $DEFAULT_ENV"
  mkdir -p "$(dirname "$HOME/.aura-env-config")"
  cat > "$HOME/.aura-env-config" << EOF
AURA_ENV="$DEFAULT_ENV"
AURA_ENV_ROOT="$AURA_ENV_ROOT"
EOF
  echo "Saved to $HOME/.aura-env-config"
fi

# ===============================
# 6. Setup complete
# ===============================

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  ✅ AURA_ENV SDK installation complete                       ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "📋 Information:"
echo "   AURA_ENV_ROOT: $AURA_ENV_ROOT"
echo "   Shell config:  $BASH_FILE"
echo ""
echo "🚀 Next steps:"
echo ""
echo "   1️⃣ Reload your shell (choose one):"
echo "      • Option A: Open a new terminal"
echo "      • Option B: Run: source $BASH_FILE"
echo ""
echo "   2️⃣ Verify the setup:"
echo "      bash scripts/env-manager.sh list"
echo ""
echo "   3️⃣ Start using:"
echo "      bash scripts/env-manager.sh select dev"
echo "      source $BASH_FILE"
echo ""
if [ -d "$AURA_ENV_ROOT/bash" ]; then
  echo "📦 Available environments:"
  for env in $(get_environments); do
    echo "      • $env"
  done
fi
echo ""
echo "📖 Documentation:"
echo "   cat $AURA_ENV_ROOT/README.md"
echo "   cat $AURA_ENV_ROOT/docs/快速开始.md"
echo ""
echo "❓ Troubleshooting:"
echo "   bash $AURA_ENV_ROOT/scripts/repair-shell-init.sh"
echo ""
