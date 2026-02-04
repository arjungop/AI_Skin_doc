#!/usr/bin/env python3
"""
Interactive A100 Setup Checklist
Walks you through the entire setup process
"""

import os
import sys
import subprocess
from pathlib import Path

GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
RED = '\033[0;31m'
BLUE = '\033[0;34m'
BOLD = '\033[1m'
NC = '\033[0m'

def clear_screen():
    os.system('clear' if os.name != 'nt' else 'cls')

def wait_for_enter(msg="Press Enter to continue..."):
    input(f"\n{YELLOW}{msg}{NC}")

def print_step(num, total, title):
    print(f"\n{BLUE}{BOLD}[Step {num}/{total}] {title}{NC}")
    print("="*60)

def run_command(cmd, check=True):
    """Run a command and return success."""
    try:
        result = subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False

def main():
    clear_screen()
    print(f"{BOLD}{'='*60}")
    print(f"  A100 SERVER SETUP CHECKLIST")
    print(f"  Complete Setup Guide for University Server")
    print(f"{'='*60}{NC}\n")
    
    print("This interactive guide will walk you through:")
    print("  1. Environment setup")
    print("  2. Dataset downloads")
    print("  3. Validation")
    print("  4. Training")
    print()
    
    wait_for_enter("Press Enter to start...")
    
    # Step 1: Kaggle API
    print_step(1, 6, "Kaggle API Setup")
    
    kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'
    if kaggle_json.exists():
        print(f"{GREEN}✅ Kaggle credentials found{NC}")
    else:
        print(f"{RED}❌ Kaggle credentials not found{NC}\n")
        print("Steps to setup:")
        print("  1. Go to: https://www.kaggle.com/settings")
        print("  2. Click 'Create New API Token'")
        print("  3. This downloads kaggle.json")
        print("  4. Run these commands:")
        print()
        print("     mkdir -p ~/.kaggle")
        print("     mv ~/Downloads/kaggle.json ~/.kaggle/")
        print("     chmod 600 ~/.kaggle/kaggle.json")
        print()
        wait_for_enter("Complete these steps and press Enter...")
        
        if not kaggle_json.exists():
            print(f"{RED}Still not found! Please complete the setup.{NC}")
            sys.exit(1)
    
    # Step 2: Run server setup
    print_step(2, 6, "Server Setup (Downloads datasets, creates environment)")
    
    print("\nThis will:")
    print("  - Create conda environment 'skindoc'")
    print("  - Install PyTorch with CUDA 12.1")
    print("  - Download ~60GB of datasets (takes time!)")
    print()
    
    response = input(f"{YELLOW}Run server setup now? (y/n): {NC}")
    if response.lower() == 'y':
        print(f"\n{BLUE}Running: bash scripts/server_setup.sh{NC}\n")
        os.system("bash scripts/server_setup.sh")
    else:
        print("\nRun manually later with:")
        print("  bash scripts/server_setup.sh")
        wait_for_enter()
    
    # Step 3: Verify conda environment
    print_step(3, 6, "Verify Conda Environment")
    
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'NOT_SET')
    print(f"\nCurrent conda environment: {conda_env}")
    
    if conda_env == 'base':
        print(f"{RED}❌ ERROR: You're in base environment!{NC}")
        print("\nRun this command:")
        print(f"{YELLOW}  conda activate skindoc{NC}")
        print("\nThen run this script again.")
        sys.exit(1)
    elif conda_env == 'skindoc':
        print(f"{GREEN}✅ Correct environment activated{NC}")
    else:
        print(f"{YELLOW}⚠️  Not in skindoc environment{NC}")
        print("\nActivate with:")
        print("  conda activate skindoc")
        wait_for_enter()
    
    # Step 4: Validation
    print_step(4, 6, "Validate Setup")
    
    print("\nRunning validation script...")
    print(f"{BLUE}python scripts/validate_setup.py{NC}\n")
    
    result = os.system("python scripts/validate_setup.py")
    
    if result != 0:
        print(f"\n{RED}Validation failed! Fix issues before continuing.{NC}")
        sys.exit(1)
    
    wait_for_enter()
    
    # Step 5: Choose training config
    print_step(5, 6, "Choose Training Configuration")
    
    configs = {
        '1': ('efficientnet_b4', 64, 50, 'Fast & Balanced (~6 hours, 85-90% acc)'),
        '2': ('convnext_large', 32, 75, 'High Accuracy (~8 hours, 87-92% acc)'),
        '3': ('swin_b', 48, 100, 'Best Results (~10 hours, 88-93% acc)'),
    }
    
    print("\nAvailable configurations:")
    for key, (backbone, batch, epochs, desc) in configs.items():
        print(f"  {key}. {desc}")
        print(f"     Backbone: {backbone}, Batch: {batch}, Epochs: {epochs}")
    
    choice = input(f"\n{YELLOW}Select configuration (1-3): {NC}")
    
    if choice not in configs:
        print(f"{RED}Invalid choice{NC}")
        sys.exit(1)
    
    backbone, batch_size, epochs, _ = configs[choice]
    
    # Step 6: Start training
    print_step(6, 6, "Start Training")
    
    cmd = f"bash scripts/run_complete_training.sh {backbone} {batch_size} {epochs}"
    
    print(f"\nConfiguration:")
    print(f"  Backbone: {backbone}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print()
    print(f"Command: {BLUE}{cmd}{NC}")
    print()
    
    response = input(f"{YELLOW}Start training now? (y/n): {NC}")
    
    if response.lower() == 'y':
        print(f"\n{GREEN}Starting training...{NC}\n")
        os.system(cmd)
    else:
        print("\nRun manually when ready:")
        print(f"  {cmd}")
    
    print(f"\n{GREEN}{'='*60}")
    print(f"  SETUP COMPLETE!")
    print(f"{'='*60}{NC}\n")
    
    print("Tips:")
    print("  - Training will save checkpoints in: checkpoints/")
    print("  - Monitor with: tail -f checkpoints/*/training.log")
    print("  - Press Ctrl+C to stop gracefully")
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{YELLOW}Setup cancelled by user{NC}")
        sys.exit(0)
