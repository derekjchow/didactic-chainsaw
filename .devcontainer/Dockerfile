FROM mcr.microsoft.com/devcontainers/base:ubuntu-22.04

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

# Install Bazel via Bazelisk
RUN curl -fSsL -o /usr/local/bin/bazelisk https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64 \
    && chmod +x /usr/local/bin/bazelisk \
    && ln -s /usr/local/bin/bazelisk /usr/local/bin/bazel

# Install Buildifier for BUILD file formatting
RUN curl -fSsL -o /usr/local/bin/buildifier https://github.com/bazelbuild/buildtools/releases/latest/download/buildifier-linux-amd64 \
    && chmod +x /usr/local/bin/buildifier

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
