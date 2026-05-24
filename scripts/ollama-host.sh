#!/usr/bin/env bash
# Print the URL the WSL viz server should use to reach a Windows-hosted Ollama.
# WSL's `localhost:11434` does NOT reach the Windows host; use the gateway IP.
gw=$(ip route show 2>/dev/null | awk '/^default/ {print $3; exit}')
echo "http://${gw:-localhost}:11434"
