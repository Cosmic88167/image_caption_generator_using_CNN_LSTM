#!/usr/bin/env python3
"""
Quick Start Script for Image Caption Generator GUI
Run this to install dependencies and start the application
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a shell command and handle errors"""
    print(f"\n{'='*60}")
    print(f"► {description}")
    print(f"{'='*60}")
    try:
        result = subprocess.run(command, shell=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error: {e}")
        return False

def main():
    print("\n")
    print("╔" + "═"*58 + "╗")
    print("║" + " "*12 + "Image Caption Generator - Quick Start" + " "*9 + "║")
    print("╚" + "═"*58 + "╝")
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("\n✗ Python 3.7+ required. You have Python", sys.version)
        sys.exit(1)
    
    print(f"\n✓ Python version: {sys.version}")
    
    # Check required files
    print("\n" + "="*60)
    print("Checking required files...")
    print("="*60)
    
    required_files = {
        "model_19.h5": "Trained caption generation model",
        "prepro_by_raj.txt": "Vocabulary and captions reference",
    }
    
    missing_files = []
    for filename, description in required_files.items():
        if os.path.exists(filename):
            print(f"✓ {filename:<30} {description}")
        else:
            print(f"✗ {filename:<30} {description}")
            missing_files.append(filename)
    
    if missing_files:
        print(f"\n⚠ Warning: Missing files: {', '.join(missing_files)}")
        response = input("Continue anyway? (y/n): ").lower()
        if response != 'y':
            sys.exit(1)
    
    # Install/upgrade pip
    if not run_command(
        f"{sys.executable} -m pip install --upgrade pip",
        "Upgrading pip..."
    ):
        print("Warning: Could not upgrade pip, continuing...")
    
    # Install requirements
    if os.path.exists("requirements.txt"):
        if not run_command(
            f"{sys.executable} -m pip install -r requirements.txt",
            "Installing dependencies from requirements.txt..."
        ):
            print("\n⚠ Some dependencies may not have installed correctly")
    else:
        print("\n⚠ requirements.txt not found, installing core packages manually...")
        packages = [
            "tensorflow>=2.10.0",
            "keras>=2.10.0",
            "gradio>=3.35.0",
            "pillow>=9.0.0",
            "numpy>=1.21.0",
            "pyttsx3>=2.90",
            "matplotlib>=3.5.0"
        ]
        if not run_command(
            f"{sys.executable} -m pip install {' '.join(packages)}",
            "Installing core packages..."
        ):
            print("✗ Failed to install packages")
            sys.exit(1)
    
    # Choose which version to run
    print("\n" + "="*60)
    print("Choose application version:")
    print("="*60)
    print("1. Full Version (with text-to-speech, advanced features)")
    print("2. Simple Version (lighter, faster, basic features)")
    print("3. Exit")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        print("\nStarting Full Version with text-to-speech...")
        run_command(f"{sys.executable} app_gui.py", "Running Full Application")
    elif choice == "2":
        print("\nStarting Simple Version...")
        run_command(f"{sys.executable} app_gui_simple.py", "Running Simple Application")
    elif choice == "3":
        print("\nExiting...")
        sys.exit(0)
    else:
        print("Invalid choice!")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        sys.exit(1)
