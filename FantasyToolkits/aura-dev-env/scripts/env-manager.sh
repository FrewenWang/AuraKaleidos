#!/bin/bash
# Environment manager for aura-dev-env

set -e

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AURA_ENV_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$HOME/.aura-env-config"

# 颜色定义
if [ -t 1 ]; then
  RED='\033[0;31m'
  GREEN='\033[0;32m'
  YELLOW='\033[1;33m'
  BLUE='\033[0;34m'
  CYAN='\033[0;36m'
  NC='\033[0m'
else
  RED='' GREEN='' YELLOW='' BLUE='' CYAN='' NC=''
fi

# ============================================================
# 工具函数
# ============================================================

print_error() {
  echo -e "${RED}❌ $*${NC}" >&2
}

print_success() {
  echo -e "${GREEN}✅ $*${NC}"
}

print_info() {
  echo -e "${BLUE}ℹ️  $*${NC}"
}

print_header() {
  echo -e "\n${CYAN}$*${NC}"
}

# 读取当前环境
get_current_env() {
  if [ -f "$CONFIG_FILE" ]; then
    grep "^AURA_ENV=" "$CONFIG_FILE" 2> /dev/null | cut -d'=' -f2 | tr -d '"' || echo "default"
  else
    echo "default"
  fi
}

# 保存环境配置
save_env_config() {
  local env_name="$1"
  mkdir -p "$(dirname "$CONFIG_FILE")"
  cat > "$CONFIG_FILE" << EOF
AURA_ENV="$env_name"
AURA_ENV_ROOT="$AURA_ENV_ROOT"
EOF
}

# 读取渠道配置（支持从配置文件或自动检测）
get_environments() {
  local config_file="$AURA_ENV_ROOT/config/environments.conf"

  # 优先从配置文件读取
  if [ -f "$config_file" ]; then
    grep -v "^#" "$config_file" | grep -v "^$" | tr '\n' ' ' | sed 's/ $//'
    return
  fi

  # 降级到从 bash/ 目录自动检测
  local bash_dir="$AURA_ENV_ROOT/bash"
  local envs=""

  if [ -d "$bash_dir" ]; then
    for dir in "$bash_dir"/*; do
      if [ -d "$dir" ]; then
        local name=$(basename "$dir")
        # 跳过特殊目录
        if [ "$name" != "base" ] && [ "$name" != "Linux" ] && [ "$name" != "Darwin" ]; then
          envs="$envs $name"
        fi
      fi
    done
  fi

  echo "$envs"
}

# 获取可用环境列表
list_environments() {
  local bash_dir="$AURA_ENV_ROOT/bash"
  local envs=$(get_environments)

  print_header "Available environments:"
  echo ""

  for env in $envs; do
    printf "  ${CYAN}%-15s${NC}" "$env"
    if [ -f "$bash_dir/$env/.description" ]; then
      head -1 "$bash_dir/$env/.description" | sed 's/^# //'
    elif [ -d "$bash_dir/$env" ]; then
      echo "Shell 配置"
    elif [ -d "$AURA_ENV_ROOT/bin/$env" ]; then
      echo "工具目录（无 Shell 配置）"
    else
      echo "（已注册，尚未配置）"
    fi
  done
}

# ============================================================
# 命令实现
# ============================================================

cmd_list() {
  list_environments
}

cmd_current() {
  local current=$(get_current_env)
  echo "Current environment: $current"
}

cmd_select() {
  local env_name="$1"

  if [ -z "$env_name" ]; then
    print_error "Usage: aura-env select <environment>"
    list_environments
    return 1
  fi

  # Environment can exist in bash/, bin/, or just be registered in config
  # It will be created on demand if directories don't exist yet
  # Just check if it's a known environment or allow dynamic environments
  local known_env=false

  # Check if registered in config
  if grep -q "^${env_name}$" "$AURA_ENV_ROOT/config/environments.conf" 2> /dev/null; then
    known_env=true
  fi

  # Check if bash directory exists
  if [ -d "$AURA_ENV_ROOT/bash/$env_name" ]; then
    known_env=true
  fi

  # Check if bin directory exists
  if [ -d "$AURA_ENV_ROOT/bin/$env_name" ]; then
    known_env=true
  fi

  if [ "$known_env" = false ]; then
    print_error "Environment not found: $env_name"
    print_info "Available environments:"
    list_environments
    return 1
  fi

  save_env_config "$env_name"
  print_success "Environment set to: $env_name"
  print_info "Reload your shell: source ~/.bashrc"
}

cmd_show() {
  local env_name="${1:-$(get_current_env)}"
  local platform=$(uname -s)
  local bash_file="$AURA_ENV_ROOT/bash/$env_name/bashrc.fwf"
  local platform_file="$AURA_ENV_ROOT/bash/$env_name/bashrc_${platform}.fwf"
  local found_config=false

  print_header "Configuration for: $env_name"
  echo ""

  # Show environment config if it exists
  if [ -f "$bash_file" ]; then
    echo "━━━ bash/$env_name/bashrc.fwf ━━━"
    cat "$bash_file"
    found_config=true
  fi

  # Show environment + platform configs if it exists
  if [ -f "$platform_file" ]; then
    if [ "$found_config" = true ]; then
      echo ""
    fi
    echo "━━━ bash/$env_name/bashrc_${platform}.fwf (platform-specific) ━━━"
    cat "$platform_file"
    found_config=true
  fi

  if [ "$found_config" = false ]; then
    print_info "No bash configuration files found for environment: $env_name"
    echo "(Environment directories may not have been created yet)"
  fi
}

cmd_validate() {
  local env_name="${1:-$(get_current_env)}"
  local platform=$(uname -s)
  local bash_file="$AURA_ENV_ROOT/bash/$env_name/bashrc.fwf"
  local platform_file="$AURA_ENV_ROOT/bash/$env_name/bashrc_${platform}.fwf"
  local all_valid=true
  local found_config=false

  print_header "Validating environment: $env_name"

  # Validate environment config if it exists
  if [ -f "$bash_file" ]; then
    found_config=true
    if bash -n "$bash_file" 2> /dev/null; then
      print_success "Environment bashrc syntax is valid"
    else
      print_error "Environment bashrc syntax error"
      all_valid=false
    fi
  fi

  # Validate environment + platform configs if it exists
  if [ -f "$platform_file" ]; then
    found_config=true
    if bash -n "$platform_file" 2> /dev/null; then
      print_success "Environment platform config (${platform}) syntax is valid"
    else
      print_error "Environment platform config (${platform}) syntax error"
      all_valid=false
    fi
  fi

  if [ "$found_config" = false ]; then
    print_info "No bash configuration files found for environment: $env_name"
    print_info "This is OK - environment can use inherited configurations"
    return 0
  fi

  if [ "$all_valid" = false ]; then
    return 1
  fi
}

cmd_info() {
  local env_name="${1:-$(get_current_env)}"
  local platform=$(uname -s)

  print_header "Environment: $env_name"
  echo ""

  # Show bash config paths (whether they exist or not)
  echo "Bash configuration paths:"
  local bash_dir="$AURA_ENV_ROOT/bash/$env_name"
  local bash_file="$AURA_ENV_ROOT/bash/$env_name/bashrc.fwf"
  local platform_file="$AURA_ENV_ROOT/bash/$env_name/bashrc_${platform}.fwf"

  if [ -d "$bash_dir" ]; then
    echo "  ✓ Directory: $bash_dir"
  else
    echo "  ○ Directory: $bash_dir (not created yet)"
  fi

  if [ -f "$bash_file" ]; then
    echo "  ✓ Config: $(basename $bash_file)"
  else
    echo "  ○ Config: $(basename $bash_file) (not found)"
  fi

  if [ -f "$platform_file" ]; then
    echo "  ✓ Platform: $(basename $platform_file) (${platform})"
  else
    echo "  ○ Platform: $(basename $platform_file) (${platform}) (not found)"
  fi

  echo ""
  echo "Bin paths:"
  local bin_dir="$AURA_ENV_ROOT/bin/$env_name"
  if [ -d "$bin_dir" ]; then
    echo "  ✓ Directory: $bin_dir"
  else
    echo "  ○ Directory: $bin_dir (not created yet)"
  fi

  local bin_platform_dir="$AURA_ENV_ROOT/bin/$env_name/$platform"
  if [ -d "$bin_platform_dir" ]; then
    echo "  ✓ Platform tools: $bin_platform_dir"
  else
    echo "  ○ Platform tools: $bin_platform_dir (not found)"
  fi
}

cmd_help() {
  cat << EOF
AURA Dev Environment Manager

USAGE:
    aura-env <command> [options]

COMMANDS:
    list                List all available environments
    current             Show current environment
    select <env>        Switch to an environment
    show [env]          Show environment configuration
    validate [env]      Validate environment setup
    info [env]          Show environment information
    help                Show this help message

EXAMPLES:
    aura-env list
    aura-env select dev
    aura-env validate
    aura-env show corporate

EOF
}

# ============================================================
# 主程序
# ============================================================

main() {
  local cmd="${1:-help}"

  case "$cmd" in
    list) cmd_list "${@:2}" ;;
    current) cmd_current ;;
    select) cmd_select "${@:2}" ;;
    show) cmd_show "${@:2}" ;;
    validate) cmd_validate "${@:2}" ;;
    info) cmd_info "${@:2}" ;;
    help | --help | -h)
      cmd_help
      ;;
    *)
      print_error "Unknown command: $cmd"
      cmd_help
      exit 1
      ;;
  esac
}

main "$@"
