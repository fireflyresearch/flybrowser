#!/usr/bin/env bash
# Copyright 2026 Firefly Software Solutions Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ============================================================================
# FlyBrowser Uninstall Script
# ============================================================================
#
# Removes FlyBrowser and all its components from the system.
#
# Usage:
#   ./uninstall.sh                    # Interactive uninstall
#   ./uninstall.sh --all              # Remove everything (including data)
#   ./uninstall.sh --keep-data        # Keep configuration and data
#   ./uninstall.sh --help             # Show help
#
# ============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m' # No Color

# Emoji support detection
if [[ "${TERM:-}" == *"256color"* ]] || [[ "${COLORTERM:-}" == "truecolor" ]]; then
    EMOJI_CHECK="✓"
    EMOJI_CROSS="✗"
    EMOJI_WARN="⚠"
    EMOJI_INFO="ℹ"
    EMOJI_TRASH="🗑"
else
    EMOJI_CHECK="[OK]"
    EMOJI_CROSS="[FAIL]"
    EMOJI_WARN="[WARN]"
    EMOJI_INFO="[INFO]"
    EMOJI_TRASH="[DEL]"
fi

# Default paths
INSTALL_DIR="/usr/local/bin"
VENV_DIR="$HOME/.flybrowser/venv"
DATA_DIR="$HOME/.flybrowser"
CONFIG_DIR="$HOME/.config/flybrowser"

# Options
REMOVE_DATA=false
KEEP_DATA=false
FORCE=false
QUIET=false

# Print functions
print_msg() {
    local color=$1
    local msg=$2
    if [ "$QUIET" = false ]; then
        echo -e "${color}${msg}${NC}"
    fi
}

print_info() { print_msg "$BLUE" "$EMOJI_INFO $1"; }
print_success() { print_msg "$GREEN" "$EMOJI_CHECK $1"; }
print_warning() { print_msg "$YELLOW" "$EMOJI_WARN $1"; }
print_error() { print_msg "$RED" "$EMOJI_CROSS $1"; }
print_step() { print_msg "$CYAN" "$EMOJI_TRASH $1"; }

# Detect OS
detect_os() {
    case "$(uname -s)" in
        Linux*)     echo "linux";;
        Darwin*)    echo "darwin";;
        CYGWIN*|MINGW*|MSYS*) echo "windows";;
        *)          echo "unknown";;
    esac
}

# Print banner
print_banner() {
    if [ "$QUIET" = false ]; then
        echo ""
        print_msg "$BLUE" '  _____.__         ___.'
        print_msg "$BLUE" '_/ ____\  | ___.__.\\_ |_________  ______  _  ________ ___________'
        print_msg "$BLUE" '\\   __\\|  |<   |  | | __ \\_  __ \\/  _ \\ \\/ \\/ /  ___// __ \\_  __ \\'
        print_msg "$BLUE" ' |  |  |  |_\\___  | | \\_\\ \\  | \\(  <_> )     /\\___ \\\\  ___/|  | \\/'
        print_msg "$BLUE" ' |__|  |____/ ____| |___  /__|   \\____/ \\/\\_//____  >\\___  >__|'
        print_msg "$BLUE" '            \\/          \\/                        \\/     \\/'
        echo ""
        print_msg "$RED" " Uninstaller"
        echo ""
    fi
}

# Show usage
show_usage() {
    cat << 'EOF'
FlyBrowser Uninstall Script

Usage: uninstall.sh [OPTIONS]

Options:
  --all             Remove everything including data and configuration
  --keep-data       Keep configuration and data (remove only binaries)
  --force, -f       Don't ask for confirmation
  --quiet, -q       Suppress output
  --help, -h        Show this help message

Examples:
  # Interactive uninstall (asks what to remove)
  ./uninstall.sh

  # Remove everything without asking
  ./uninstall.sh --all --force

  # Remove only binaries, keep config
  ./uninstall.sh --keep-data

Paths that may be removed:
  CLI Commands:     /usr/local/bin/flybrowser*
  Virtual Env:      ~/.flybrowser/venv
  Data Directory:   ~/.flybrowser
  Config Directory: ~/.config/flybrowser
  macOS Service:    ~/Library/LaunchAgents/dev.flybrowser.plist
  Linux Service:    /etc/systemd/system/flybrowser.service
EOF
}

# Parse arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --all)
                REMOVE_DATA=true
                shift
                ;;
            --keep-data)
                KEEP_DATA=true
                shift
                ;;
            --force|-f)
                FORCE=true
                shift
                ;;
            --quiet|-q)
                QUIET=true
                shift
                ;;
            --help|-h)
                show_usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                echo ""
                show_usage
                exit 1
                ;;
        esac
    done
}

# Ask for confirmation
confirm() {
    local message=$1
    local default=${2:-n}
    
    if [ "$FORCE" = true ]; then
        return 0
    fi
    
    local prompt
    if [ "$default" = "y" ]; then
        prompt="[Y/n]"
    else
        prompt="[y/N]"
    fi
    
    read -p "$message $prompt " -n 1 -r reply
    echo
    
    if [ -z "$reply" ]; then
        reply=$default
    fi
    
    [[ "$reply" =~ ^[Yy]$ ]]
}

# Stop and remove services
remove_services() {
    local os=$(detect_os)
    local removed=false
    
    print_step "Checking for services..."
    
    # macOS launchd
    if [ "$os" = "darwin" ]; then
        local plist_path="$HOME/Library/LaunchAgents/dev.flybrowser.plist"
        if [ -f "$plist_path" ]; then
            print_info "Found launchd service"
            launchctl unload "$plist_path" 2>/dev/null || true
            rm -f "$plist_path"
            print_success "Removed launchd service"
            removed=true
        fi
    fi
    
    # Linux systemd
    if [ "$os" = "linux" ]; then
        if [ -f "/etc/systemd/system/flybrowser.service" ]; then
            print_info "Found systemd service"
            sudo systemctl stop flybrowser 2>/dev/null || true
            sudo systemctl disable flybrowser 2>/dev/null || true
            sudo rm -f /etc/systemd/system/flybrowser.service
            sudo systemctl daemon-reload
            print_success "Removed systemd service"
            removed=true
        fi
    fi
    
    if [ "$removed" = false ]; then
        print_info "No services found"
    fi
}

# Remove CLI wrappers
remove_cli_wrappers() {
    print_step "Removing CLI commands..."
    
    local wrappers=(
        "flybrowser"
        "flybrowser-serve"
        "flybrowser-setup"
        "flybrowser-cluster"
        "flybrowser-admin"
        "flybrowser-uninstall"
    )
    
    local removed=0
    
    for wrapper in "${wrappers[@]}"; do
        local path="$INSTALL_DIR/$wrapper"
        if [ -f "$path" ]; then
            if [ -w "$INSTALL_DIR" ]; then
                rm -f "$path"
            else
                sudo rm -f "$path"
            fi
            print_success "Removed $path"
            ((removed++))
        fi
    done
    
    if [ $removed -eq 0 ]; then
        print_info "No CLI commands found in $INSTALL_DIR"
    else
        print_success "Removed $removed CLI command(s)"
    fi
}

# Remove virtual environment
remove_venv() {
    print_step "Removing virtual environment..."
    
    if [ -d "$VENV_DIR" ]; then
        rm -rf "$VENV_DIR"
        print_success "Removed $VENV_DIR"
    else
        print_info "Virtual environment not found at $VENV_DIR"
    fi
}

# Remove data and configuration
remove_data() {
    print_step "Removing data and configuration..."
    
    local removed=false
    
    # Remove data directory
    if [ -d "$DATA_DIR" ]; then
        rm -rf "$DATA_DIR"
        print_success "Removed $DATA_DIR"
        removed=true
    fi
    
    # Remove config directory
    if [ -d "$CONFIG_DIR" ]; then
        rm -rf "$CONFIG_DIR"
        print_success "Removed $CONFIG_DIR"
        removed=true
    fi
    
    # Remove .env in current directory if it's a flybrowser config
    if [ -f ".env" ] && grep -q "FLYBROWSER" ".env" 2>/dev/null; then
        if confirm "Remove .env file in current directory?"; then
            rm -f ".env"
            print_success "Removed .env"
            removed=true
        fi
    fi
    
    if [ "$removed" = false ]; then
        print_info "No data directories found"
    fi
}

# Print summary
print_summary() {
    echo ""
    echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
    print_msg "$GREEN" "$EMOJI_CHECK FlyBrowser has been uninstalled"
    echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
    echo ""
    
    if [ "$KEEP_DATA" = true ] || [ "$REMOVE_DATA" = false ]; then
        print_info "Configuration and data were preserved"
        print_info "To reinstall: curl -fsSL https://get.flybrowser.dev | bash"
    else
        print_info "All FlyBrowser files have been removed"
    fi
    echo ""
    
    print_info "To reinstall FlyBrowser:"
    echo "  curl -fsSL https://get.flybrowser.dev | bash"
    echo "  # or"
    echo "  git clone https://github.com/firefly-oss/flybrowser && cd flybrowser && ./install.sh"
    echo ""
}

# Main uninstall flow
main() {
    parse_args "$@"
    
    print_banner
    
    # Check if FlyBrowser is installed
    local installed=false
    if [ -d "$VENV_DIR" ] || [ -f "$INSTALL_DIR/flybrowser" ]; then
        installed=true
    fi
    
    if [ "$installed" = false ]; then
        print_warning "FlyBrowser does not appear to be installed"
        print_info "Checked: $VENV_DIR"
        print_info "Checked: $INSTALL_DIR/flybrowser"
        exit 0
    fi
    
    # Show what will be removed
    echo ""
    print_msg "$BOLD" "The following will be removed:"
    echo ""
    
    if [ -f "$INSTALL_DIR/flybrowser" ]; then
        echo "  • CLI commands in $INSTALL_DIR"
    fi
    
    if [ -d "$VENV_DIR" ]; then
        echo "  • Virtual environment: $VENV_DIR"
    fi
    
    local os=$(detect_os)
    if [ "$os" = "darwin" ] && [ -f "$HOME/Library/LaunchAgents/dev.flybrowser.plist" ]; then
        echo "  • macOS launchd service"
    fi
    if [ "$os" = "linux" ] && [ -f "/etc/systemd/system/flybrowser.service" ]; then
        echo "  • Linux systemd service"
    fi
    
    if [ "$KEEP_DATA" = false ]; then
        if [ "$REMOVE_DATA" = true ]; then
            if [ -d "$DATA_DIR" ]; then
                echo "  • Data directory: $DATA_DIR"
            fi
            if [ -d "$CONFIG_DIR" ]; then
                echo "  • Config directory: $CONFIG_DIR"
            fi
        else
            echo ""
            print_msg "$DIM" "  (Data and configuration will be preserved unless --all is used)"
        fi
    fi
    
    echo ""
    
    # Confirm uninstall
    if ! confirm "Proceed with uninstall?"; then
        print_info "Uninstall cancelled"
        exit 0
    fi
    
    echo ""
    
    # Perform uninstall
    remove_services
    remove_cli_wrappers
    remove_venv
    
    # Handle data removal
    if [ "$KEEP_DATA" = false ]; then
        if [ "$REMOVE_DATA" = true ]; then
            remove_data
        elif [ "$FORCE" = false ]; then
            echo ""
            if confirm "Also remove data and configuration (~/.flybrowser)?"; then
                remove_data
            else
                print_info "Keeping data and configuration"
            fi
        fi
    fi
    
    print_summary
}

# Run main
main "$@"
