#!/bin/bash
set -e

# Ensure .claudelocal mount target exists on the host
# (the bind mount will create it, but permissions may need fixing)
if [ -d "$HOME/.claude" ]; then
  echo "Claude config directory found at $HOME/.claude"
fi

# Install Gemini CLI
echo "Installing Gemini CLI..."
npm install -g @google/gemini-cli

# Set up Python environment with uv
echo "Setting up Python environment with uv..."
uv python install 3.12
uv init --no-workspace --python 3.12 2>/dev/null || true
uv add tensorflow jax

# Pre-fetch Bazel dependencies (optional, speeds up first build)
echo "Fetching Bazel dependencies..."
bazel fetch //src:main 2>/dev/null || echo "Bazel fetch skipped (run manually with: bazel fetch //src:main)"

echo "Dev container setup complete!"
