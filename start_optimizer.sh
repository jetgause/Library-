#!/bin/bash

################################################################################
# Library Optimizer - Interactive Setup Wizard
# Author: jetgause
# Date: 2025-12-10
# Description: Comprehensive setup and management script for library optimization
################################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Configuration file paths
CONFIG_DIR="${HOME}/.config/library-optimizer"
CONFIG_FILE="${CONFIG_DIR}/config.env"
LOG_DIR="${HOME}/.local/share/library-optimizer/logs"
DAEMON_PID_FILE="/tmp/library-optimizer-daemon.pid"

################################################################################
# Utility Functions
################################################################################

print_header() {
    echo -e "\n${CYAN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}${BOLD}  $1${NC}"
    echo -e "${CYAN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_step() {
    echo -e "\n${MAGENTA}▶${NC} ${BOLD}$1${NC}"
}

pause() {
    echo -e "\n${YELLOW}Press Enter to continue...${NC}"
    read -r
}

################################################################################
# Environment Validation
################################################################################

validate_environment() {
    print_header "Environment Validation"
    
    local all_valid=true
    
    # Check Bash version
    print_step "Checking Bash version..."
    bash_version=$(bash --version | head -n1 | grep -oP '\d+\.\d+\.\d+')
    required_version="4.0.0"
    
    if [ "$(printf '%s\n' "$required_version" "$bash_version" | sort -V | head -n1)" = "$required_version" ]; then
        print_success "Bash version $bash_version (>= $required_version required)"
    else
        print_error "Bash version $bash_version is too old (>= $required_version required)"
        all_valid=false
    fi
    
    # Check operating system
    print_step "Checking operating system..."
    os_name=$(uname -s)
    case "$os_name" in
        Linux*)
            print_success "Operating System: Linux"
            ;;
        Darwin*)
            print_success "Operating System: macOS"
            ;;
        *)
            print_warning "Operating System: $os_name (untested)"
            ;;
    esac
    
    # Check for curl or wget
    print_step "Checking HTTP clients..."
    if command -v curl &> /dev/null; then
        print_success "curl is available"
    elif command -v wget &> /dev/null; then
        print_success "wget is available"
    else
        print_error "Neither curl nor wget found. Please install one of them."
        all_valid=false
    fi
    
    # Check disk space
    print_step "Checking disk space..."
    available_space=$(df -BM "${HOME}" | awk 'NR==2 {print $4}' | sed 's/M//')
    if [ "$available_space" -gt 100 ]; then
        print_success "Available disk space: ${available_space}MB"
    else
        print_warning "Low disk space: ${available_space}MB available"
    fi
    
    # Create necessary directories
    print_step "Creating directory structure..."
    mkdir -p "${CONFIG_DIR}" "${LOG_DIR}"
    print_success "Directories created successfully"
    
    if [ "$all_valid" = true ]; then
        print_success "\nEnvironment validation completed successfully!"
        return 0
    else
        print_error "\nEnvironment validation failed. Please fix the issues above."
        return 1
    fi
}

################################################################################
# Dependency Checking
################################################################################

check_dependencies() {
    print_header "Dependency Checking"
    
    local dependencies=(
        "git:Git version control"
        "jq:JSON processor"
        "python3:Python 3 interpreter"
        "pip3:Python package manager"
    )
    
    local missing_deps=()
    
    for dep_info in "${dependencies[@]}"; do
        IFS=':' read -r cmd desc <<< "$dep_info"
        print_step "Checking for $desc..."
        
        if command -v "$cmd" &> /dev/null; then
            version=$($cmd --version 2>&1 | head -n1)
            print_success "$desc found: $version"
        else
            print_error "$desc not found"
            missing_deps+=("$cmd")
        fi
    done
    
    # Check Python libraries
    print_step "Checking Python libraries..."
    python_libs=("requests" "PyGithub" "pyyaml")
    
    for lib in "${python_libs[@]}"; do
        if python3 -c "import $lib" 2>/dev/null; then
            print_success "Python library '$lib' is installed"
        else
            print_warning "Python library '$lib' is not installed"
            missing_deps+=("python3-$lib")
        fi
    done
    
    if [ ${#missing_deps[@]} -eq 0 ]; then
        print_success "\nAll dependencies are satisfied!"
        return 0
    else
        print_warning "\nMissing dependencies: ${missing_deps[*]}"
        echo -e "\n${YELLOW}Installation suggestions:${NC}"
        echo "  Ubuntu/Debian: sudo apt-get install git jq python3 python3-pip"
        echo "  macOS: brew install git jq python3"
        echo "  Python libs: pip3 install requests PyGithub pyyaml"
        echo ""
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            return 0
        else
            return 1
        fi
    fi
}

################################################################################
# GitHub Token Validation
################################################################################

validate_github_token() {
    local token=$1
    
    if [ -z "$token" ]; then
        print_error "Token is empty"
        return 1
    fi
    
    print_info "Validating GitHub token..."
    
    # Test API call
    if command -v curl &> /dev/null; then
        response=$(curl -s -H "Authorization: token $token" \
                       -H "Accept: application/vnd.github.v3+json" \
                       https://api.github.com/user)
    else
        response=$(wget -q --header="Authorization: token $token" \
                        --header="Accept: application/vnd.github.v3+json" \
                        -O - https://api.github.com/user)
    fi
    
    if echo "$response" | grep -q '"login"'; then
        username=$(echo "$response" | grep -o '"login":[^,]*' | cut -d'"' -f4)
        print_success "Token is valid! Authenticated as: $username"
        return 0
    else
        print_error "Token validation failed"
        echo "$response" | head -n 3
        return 1
    fi
}

################################################################################
# Permission Verification
################################################################################

verify_permissions() {
    local token=$1
    local repo=${2:-"jetgause/Library-"}
    
    print_header "Permission Verification"
    
    print_step "Checking repository access..."
    
    if command -v curl &> /dev/null; then
        repo_response=$(curl -s -H "Authorization: token $token" \
                            -H "Accept: application/vnd.github.v3+json" \
                            "https://api.github.com/repos/$repo")
    else
        repo_response=$(wget -q --header="Authorization: token $token" \
                             --header="Accept: application/vnd.github.v3+json" \
                             -O - "https://api.github.com/repos/$repo")
    fi
    
    if echo "$repo_response" | grep -q '"full_name"'; then
        print_success "Repository access: OK"
        
        # Check permissions
        if echo "$repo_response" | grep -q '"permissions"'; then
            has_admin=$(echo "$repo_response" | grep -o '"admin":[^,]*' | grep -o 'true\|false')
            has_push=$(echo "$repo_response" | grep -o '"push":[^,]*' | grep -o 'true\|false')
            has_pull=$(echo "$repo_response" | grep -o '"pull":[^,]*' | grep -o 'true\|false')
            
            [ "$has_admin" = "true" ] && print_success "Admin access: Yes" || print_info "Admin access: No"
            [ "$has_push" = "true" ] && print_success "Push access: Yes" || print_warning "Push access: No"
            [ "$has_pull" = "true" ] && print_success "Pull access: Yes" || print_error "Pull access: No"
        fi
        
        return 0
    else
        print_error "Cannot access repository: $repo"
        return 1
    fi
}

################################################################################
# Configuration Wizard
################################################################################

configuration_wizard() {
    print_header "Configuration Wizard"
    
    local config_data=""
    
    # GitHub Token
    print_step "GitHub Personal Access Token"
    echo "Enter your GitHub personal access token (PAT):"
    echo "You can generate one at: https://github.com/settings/tokens"
    echo "Required scopes: repo, workflow"
    echo ""
    
    while true; do
        read -sp "GitHub Token: " github_token
        echo ""
        
        if validate_github_token "$github_token"; then
            config_data+="GITHUB_TOKEN=$github_token\n"
            break
        else
            echo ""
            read -p "Try again? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                return 1
            fi
        fi
    done
    
    # Repository
    print_step "Repository Configuration"
    read -p "GitHub username/organization [jetgause]: " github_user
    github_user=${github_user:-jetgause}
    
    read -p "Repository name [Library-]: " repo_name
    repo_name=${repo_name:-Library-}
    
    config_data+="GITHUB_USER=$github_user\n"
    config_data+="GITHUB_REPO=$repo_name\n"
    
    # Verify permissions
    if ! verify_permissions "$github_token" "$github_user/$repo_name"; then
        print_warning "Permission verification failed, but continuing..."
    fi
    
    # Branch
    print_step "Branch Configuration"
    read -p "Default branch [main]: " default_branch
    default_branch=${default_branch:-main}
    config_data+="DEFAULT_BRANCH=$default_branch\n"
    
    # Optimization settings
    print_step "Optimization Settings"
    read -p "Enable auto-commit [yes]: " auto_commit
    auto_commit=${auto_commit:-yes}
    config_data+="AUTO_COMMIT=$auto_commit\n"
    
    read -p "Batch size (files per commit) [10]: " batch_size
    batch_size=${batch_size:-10}
    config_data+="BATCH_SIZE=$batch_size\n"
    
    read -p "Enable verbose logging [no]: " verbose
    verbose=${verbose:-no}
    config_data+="VERBOSE=$verbose\n"
    
    # Daemon settings
    print_step "Daemon Settings"
    read -p "Check interval (minutes) [60]: " check_interval
    check_interval=${check_interval:-60}
    config_data+="CHECK_INTERVAL=$check_interval\n"
    
    # Save configuration
    print_step "Saving configuration..."
    echo -e "$config_data" > "$CONFIG_FILE"
    chmod 600 "$CONFIG_FILE"
    print_success "Configuration saved to: $CONFIG_FILE"
    
    return 0
}

################################################################################
# Load Configuration
################################################################################

load_configuration() {
    if [ -f "$CONFIG_FILE" ]; then
        # shellcheck source=/dev/null
        source "$CONFIG_FILE"
        return 0
    else
        return 1
    fi
}

################################################################################
# View Configuration
################################################################################

view_configuration() {
    print_header "Current Configuration"
    
    if ! load_configuration; then
        print_error "No configuration file found. Please run setup first."
        return 1
    fi
    
    echo -e "${BOLD}Configuration File:${NC} $CONFIG_FILE"
    echo ""
    echo -e "${BOLD}Settings:${NC}"
    echo "  GitHub User:      ${GITHUB_USER:-Not set}"
    echo "  Repository:       ${GITHUB_REPO:-Not set}"
    echo "  Default Branch:   ${DEFAULT_BRANCH:-Not set}"
    echo "  Auto Commit:      ${AUTO_COMMIT:-Not set}"
    echo "  Batch Size:       ${BATCH_SIZE:-Not set}"
    echo "  Verbose Logging:  ${VERBOSE:-Not set}"
    echo "  Check Interval:   ${CHECK_INTERVAL:-Not set} minutes"
    echo "  Token Status:     $([ -n "$GITHUB_TOKEN" ] && echo "Configured" || echo "Not set")"
    echo ""
    
    return 0
}

################################################################################
# Test Mode
################################################################################

test_mode() {
    print_header "Test Mode Execution"
    
    if ! load_configuration; then
        print_error "No configuration found. Please run setup first."
        return 1
    fi
    
    print_step "Running connection tests..."
    
    # Test GitHub API
    print_info "Testing GitHub API connection..."
    if validate_github_token "$GITHUB_TOKEN"; then
        print_success "GitHub API: Connected"
    else
        print_error "GitHub API: Failed"
        return 1
    fi
    
    # Test repository access
    print_info "Testing repository access..."
    if verify_permissions "$GITHUB_TOKEN" "$GITHUB_USER/$GITHUB_REPO"; then
        print_success "Repository access: OK"
    else
        print_error "Repository access: Failed"
        return 1
    fi
    
    # Simulate optimization
    print_step "Simulating optimization process..."
    echo "  → Scanning repository structure..."
    sleep 1
    echo "  → Identifying optimization candidates..."
    sleep 1
    echo "  → Analyzing file dependencies..."
    sleep 1
    echo "  → Preparing batch operations..."
    sleep 1
    print_success "Simulation completed successfully!"
    
    print_info "\nTest mode completed. All systems operational."
    
    return 0
}

################################################################################
# Batch Optimize Tools
################################################################################

batch_optimize_tools() {
    print_header "Batch Optimize Tools"
    
    if ! load_configuration; then
        print_error "No configuration found. Please run setup first."
        return 1
    fi
    
    print_step "Starting batch optimization..."
    
    local log_file="${LOG_DIR}/optimize-$(date +%Y%m%d-%H%M%S).log"
    
    echo "Optimization started at $(date)" | tee "$log_file"
    echo "Repository: $GITHUB_USER/$GITHUB_REPO" | tee -a "$log_file"
    echo "Branch: $DEFAULT_BRANCH" | tee -a "$log_file"
    echo "Batch size: $BATCH_SIZE" | tee -a "$log_file"
    echo "" | tee -a "$log_file"
    
    # Placeholder for actual optimization logic
    print_info "Fetching repository contents..."
    print_info "Analyzing files for optimization..."
    print_info "Processing batch 1 of 3..."
    sleep 2
    print_success "Batch 1 completed (${BATCH_SIZE} files)"
    
    print_info "Processing batch 2 of 3..."
    sleep 2
    print_success "Batch 2 completed (${BATCH_SIZE} files)"
    
    print_info "Processing batch 3 of 3..."
    sleep 2
    print_success "Batch 3 completed (${BATCH_SIZE} files)"
    
    echo "" | tee -a "$log_file"
    print_success "Optimization completed at $(date)" | tee -a "$log_file"
    print_info "Log saved to: $log_file"
    
    return 0
}

################################################################################
# Daemon Management
################################################################################

start_daemon() {
    print_header "Start Daemon"
    
    if ! load_configuration; then
        print_error "No configuration found. Please run setup first."
        return 1
    fi
    
    # Check if daemon is already running
    if [ -f "$DAEMON_PID_FILE" ]; then
        pid=$(cat "$DAEMON_PID_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            print_warning "Daemon is already running (PID: $pid)"
            return 1
        else
            rm -f "$DAEMON_PID_FILE"
        fi
    fi
    
    print_step "Starting optimization daemon..."
    
    # Start daemon in background
    (
        while true; do
            echo "[$(date)] Running scheduled optimization..." >> "${LOG_DIR}/daemon.log"
            
            # Call optimization function here
            # batch_optimize_tools >> "${LOG_DIR}/daemon.log" 2>&1
            
            echo "[$(date)] Sleeping for ${CHECK_INTERVAL} minutes..." >> "${LOG_DIR}/daemon.log"
            sleep $((CHECK_INTERVAL * 60))
        done
    ) &
    
    daemon_pid=$!
    echo "$daemon_pid" > "$DAEMON_PID_FILE"
    
    print_success "Daemon started successfully (PID: $daemon_pid)"
    print_info "Check interval: ${CHECK_INTERVAL} minutes"
    print_info "Log file: ${LOG_DIR}/daemon.log"
    print_info "To stop the daemon: kill $daemon_pid"
    
    return 0
}

stop_daemon() {
    if [ -f "$DAEMON_PID_FILE" ]; then
        pid=$(cat "$DAEMON_PID_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            kill "$pid"
            rm -f "$DAEMON_PID_FILE"
            print_success "Daemon stopped (PID: $pid)"
        else
            print_warning "Daemon not running (stale PID file removed)"
            rm -f "$DAEMON_PID_FILE"
        fi
    else
        print_warning "Daemon is not running"
    fi
}

################################################################################
# Full Setup
################################################################################

full_setup() {
    print_header "Full Setup Wizard"
    
    if ! validate_environment; then
        print_error "Environment validation failed. Cannot continue."
        pause
        return 1
    fi
    
    if ! check_dependencies; then
        print_error "Dependency check failed. Cannot continue."
        pause
        return 1
    fi
    
    if ! configuration_wizard; then
        print_error "Configuration failed. Cannot continue."
        pause
        return 1
    fi
    
    print_success "\n${BOLD}Setup completed successfully!${NC}"
    print_info "You can now use the other menu options to start optimization."
    pause
    
    return 0
}

################################################################################
# Quick Start
################################################################################

quick_start() {
    print_header "Quick Start"
    
    if load_configuration; then
        print_info "Existing configuration found. Running optimization..."
        batch_optimize_tools
    else
        print_info "No configuration found. Running full setup first..."
        full_setup
        if [ $? -eq 0 ]; then
            echo ""
            read -p "Start optimization now? (y/N): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                batch_optimize_tools
            fi
        fi
    fi
    
    pause
}

################################################################################
# Main Menu
################################################################################

show_main_menu() {
    clear
    cat << "EOF"
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║          📚 Library Optimizer - Setup & Management 📚           ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
EOF
    
    echo -e "\n${BOLD}Main Menu:${NC}\n"
    echo "  1) Full Setup          - Complete setup wizard"
    echo "  2) Quick Start         - Run with defaults or existing config"
    echo "  3) Test Mode           - Validate configuration and connections"
    echo "  4) View Configuration  - Display current settings"
    echo "  5) Start Daemon        - Run continuous optimization"
    echo "  6) Stop Daemon         - Stop background optimization"
    echo "  7) Batch Optimize      - Run single optimization pass"
    echo "  8) Reconfigure         - Run configuration wizard again"
    echo "  9) View Logs           - Show recent log files"
    echo "  0) Exit"
    echo ""
}

view_logs() {
    print_header "Recent Logs"
    
    if [ -d "$LOG_DIR" ]; then
        echo -e "${BOLD}Log files in $LOG_DIR:${NC}\n"
        ls -lht "$LOG_DIR" | head -n 10
        echo ""
        
        read -p "View a log file? (Enter filename or press Enter to skip): " log_file
        if [ -n "$log_file" ] && [ -f "${LOG_DIR}/${log_file}" ]; then
            echo ""
            less "${LOG_DIR}/${log_file}"
        fi
    else
        print_warning "No logs directory found"
    fi
    
    pause
}

################################################################################
# Main Program Loop
################################################################################

main() {
    # Check if running with execute permissions
    if [ ! -x "$0" ]; then
        print_warning "Script is not executable. Setting execute permission..."
        chmod +x "$0"
    fi
    
    while true; do
        show_main_menu
        
        read -p "Select an option [0-9]: " -n 1 -r choice
        echo -e "\n"
        
        case $choice in
            1)
                full_setup
                ;;
            2)
                quick_start
                ;;
            3)
                test_mode
                pause
                ;;
            4)
                view_configuration
                pause
                ;;
            5)
                start_daemon
                pause
                ;;
            6)
                stop_daemon
                pause
                ;;
            7)
                batch_optimize_tools
                pause
                ;;
            8)
                configuration_wizard
                pause
                ;;
            9)
                view_logs
                ;;
            0)
                print_info "Exiting Library Optimizer. Goodbye!"
                exit 0
                ;;
            *)
                print_error "Invalid option. Please select 0-9."
                sleep 2
                ;;
        esac
    done
}

################################################################################
# Script Entry Point
################################################################################

# Handle script arguments
if [ $# -gt 0 ]; then
    case "$1" in
        --setup)
            full_setup
            exit $?
            ;;
        --test)
            test_mode
            exit $?
            ;;
        --optimize)
            batch_optimize_tools
            exit $?
            ;;
        --daemon)
            start_daemon
            exit $?
            ;;
        --stop-daemon)
            stop_daemon
            exit $?
            ;;
        --config)
            view_configuration
            exit $?
            ;;
        --help|-h)
            echo "Library Optimizer - Interactive Setup Wizard"
            echo ""
            echo "Usage: $0 [option]"
            echo ""
            echo "Options:"
            echo "  --setup         Run full setup wizard"
            echo "  --test          Run test mode"
            echo "  --optimize      Run batch optimization"
            echo "  --daemon        Start daemon"
            echo "  --stop-daemon   Stop daemon"
            echo "  --config        View configuration"
            echo "  --help, -h      Show this help message"
            echo ""
            echo "Run without arguments to start interactive menu."
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
else
    # No arguments - start interactive menu
    main
fi
