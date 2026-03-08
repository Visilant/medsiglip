#!/usr/bin/env bash
# Setup script for MedSigLIP fine-tuning experiments
# Uses uv for dependency management in an isolated venv
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Sync dependencies (creates .venv if needed)
echo "Syncing dependencies..."
uv sync

# Verify GPU access
echo "Checking GPU access..."
.venv/bin/python -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    free = torch.cuda.mem_get_info(i)[0] / 1e9
    print(f'  GPU {i}: {props.name}, {props.total_mem / 1e9:.1f}GB total, {free:.1f}GB free')
"

# Verify transformers can load the model config
echo "Checking model access..."
.venv/bin/python -c "
from transformers import AutoConfig
config = AutoConfig.from_pretrained('google/medsiglip-448')
print(f'Model config loaded: {config.model_type}')
"

# Check wandb login
echo "Checking W&B..."
.venv/bin/python -c "
import wandb
if wandb.api.api_key:
    print('W&B authenticated')
else:
    print('WARNING: W&B not authenticated. Run: .venv/bin/wandb login')
"

echo ""
echo "Setup complete. Activate with: source .venv/bin/activate"
echo "Or run scripts with: .venv/bin/python <script>"
