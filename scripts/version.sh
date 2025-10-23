#!/bin/bash

# Nitrate version management script
# Usage: ./scripts/version.sh [get|set|show] [VERSION]

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

function get_current_version() {
    local cargo_toml="$WORKSPACE_ROOT/bins/nitrate-cli/Cargo.toml"
    if [[ -f "$cargo_toml" ]]; then
        grep '^version' "$cargo_toml" | head -1 | sed 's/.*"\(.*\)".*/\1/'
    else
        echo "0.1.0"
    fi
}

function set_version() {
    local new_version="$1"

    if [[ -z "$new_version" ]]; then
        echo -e "${RED}Error: Version not specified${NC}"
        echo "Usage: $0 set <version>"
        exit 1
    fi

    # Validate version format (basic check)
    if ! [[ "$new_version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        echo -e "${RED}Error: Invalid version format. Expected: X.Y.Z${NC}"
        exit 1
    fi

    echo -e "${GREEN}Setting version to $new_version across workspace...${NC}"

    # Update all Cargo.toml files
    find "$WORKSPACE_ROOT" -name "Cargo.toml" -type f | while read -r toml; do
        if grep -q '^version = "' "$toml"; then
            # Update the version line
            sed -i.bak "s/^version = \".*\"/version = \"$new_version\"/" "$toml"

            # Also update path dependencies to workspace crates
            sed -i.bak "s/\(nitrate-[a-z-]*\s*=\s*{\s*path\s*=.*version\s*=\s*\)\"[^\"]*\"/\1\"$new_version\"/" "$toml"

            # Clean up backup files
            rm -f "${toml}.bak"

            # Show which file was updated
            relative_path="${toml#$WORKSPACE_ROOT/}"
            echo "  Updated: $relative_path"
        fi
    done

    echo -e "${GREEN}Version set to $new_version${NC}"
}

function show_versions() {
    echo -e "${YELLOW}Workspace crate versions:${NC}"
    echo ""

    find "$WORKSPACE_ROOT" -name "Cargo.toml" -type f | sort | while read -r toml; do
        if grep -q '^name = "nitrate' "$toml"; then
            crate_name=$(grep '^name = "' "$toml" | head -1 | sed 's/.*"\(.*\)".*/\1/')
            version=$(grep '^version = "' "$toml" | head -1 | sed 's/.*"\(.*\)".*/\1/')
            relative_path="${toml#$WORKSPACE_ROOT/}"
            printf "  %-25s %s\n" "$crate_name:" "$version"
        fi
    done
}

function bump_version() {
    local bump_type="$1"
    local current_version=$(get_current_version)

    IFS='.' read -r major minor patch <<< "$current_version"

    case "$bump_type" in
        major)
            major=$((major + 1))
            minor=0
            patch=0
            ;;
        minor)
            minor=$((minor + 1))
            patch=0
            ;;
        patch)
            patch=$((patch + 1))
            ;;
        *)
            echo -e "${RED}Error: Invalid bump type. Use: major, minor, or patch${NC}"
            exit 1
            ;;
    esac

    new_version="${major}.${minor}.${patch}"
    echo -e "${GREEN}Bumping from $current_version to $new_version${NC}"
    set_version "$new_version"
}

function print_usage() {
    cat << EOF
Nitrate Version Management Tool

Usage: $0 <command> [arguments]

Commands:
    get                 Get the current workspace version
    set <version>       Set version across all workspace crates
    show                Show versions of all workspace crates
    bump <type>         Bump version (major|minor|patch)
    help                Show this help message

Examples:
    $0 get              # Shows current version
    $0 set 1.2.3        # Sets all crates to version 1.2.3
    $0 show             # Lists all crate versions
    $0 bump patch       # Bumps patch version (e.g., 1.2.3 -> 1.2.4)
    $0 bump minor       # Bumps minor version (e.g., 1.2.3 -> 1.3.0)
    $0 bump major       # Bumps major version (e.g., 1.2.3 -> 2.0.0)

EOF
}

# Main script logic
case "$1" in
    get)
        version=$(get_current_version)
        echo "$version"
        ;;
    set)
        set_version "$2"
        ;;
    show)
        show_versions
        ;;
    bump)
        bump_version "$2"
        ;;
    help|--help|-h)
        print_usage
        ;;
    *)
        if [[ -z "$1" ]]; then
            print_usage
        else
            echo -e "${RED}Error: Unknown command '$1'${NC}"
            echo ""
            print_usage
            exit 1
        fi
        ;;
esac
