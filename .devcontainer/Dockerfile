FROM ubuntu:24.04

# Install dependencies for Bazel and C++ builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    clang \
    curl \
    git \
    openjdk-21-jdk-headless \
    python3 \
    python3-numpy \
    zip \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Bazelisk (provides Bazel, respects .bazelversion)
RUN curl -fsSL -o /usr/local/bin/bazel \
    https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64 \
    && chmod +x /usr/local/bin/bazel

# Gemini CLI is installed after Node.js is set up by devcontainer features

# Install Buildifier for BUILD file formatting
RUN curl -fSsL -o /usr/local/bin/buildifier https://github.com/bazelbuild/buildtools/releases/latest/download/buildifier-linux-amd64 \
    && chmod +x /usr/local/bin/buildifier

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

# Create non-root user (replace ubuntu user that owns UID/GID 1000 in 24.04)
RUN userdel -r ubuntu 2>/dev/null || true \
    && groupadd --gid 1000 vscode 2>/dev/null || groupmod -n vscode $(getent group 1000 | cut -d: -f1) \
    && useradd --uid 1000 --gid 1000 -m -s /bin/bash vscode

USER vscode

# Install Claude Code
ENV PATH="/home/vscode/.local/bin:${PATH}"
RUN curl -fsSL https://claude.ai/install.sh | bash