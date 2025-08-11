#!/bin/bash

# Script to build and publish latteries package to PyPI using uv

set -e  # Exit on any error

echo "🚀 Building and publishing latteries package to PyPI"

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info/

# Build the package with uv
echo "🔨 Building the package with uv..."
uv build

# Publish to PyPI with uv
echo "📤 Publishing to PyPI..."
echo "Note: Set UV_PUBLISH_TOKEN environment variable with your PyPI token"
echo "Or use --token flag. Get token at: https://pypi.org/manage/account/token/"
uv publish

echo "✅ Package published successfully!"
echo "📦 Install with: pip install latteries"