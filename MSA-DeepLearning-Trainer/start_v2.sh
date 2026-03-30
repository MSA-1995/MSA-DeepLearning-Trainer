#!/bin/bash
# 🧠 MSA Deep Learning Trainer V2 - Start Script

echo "================================================="
echo " Starting MSA Deep Learning Trainer v2 (Local)"
echo "================================================="
echo ""

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Run the training script
python "$SCRIPT_DIR/core/deep_trainer_v2.py"

echo ""
echo "================================================="
echo " Training script has finished."
echo "================================================="
