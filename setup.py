#!/usr/bin/env python3
"""
Setup script for Medical AI Demos with Gemini 2.5 Flash
"""

import sys
import subprocess
import os

def check_python_version():
    """Check if Python version is 3.8 or higher."""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required.")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    else:
        print(f"✅ Python version: {sys.version.split()[0]}")

def install_requirements():
    """Install required packages."""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
    except subprocess.CalledProcessError:
        print("❌ Error installing dependencies. Please install manually:")
        print("pip install -r requirements.txt")
        sys.exit(1)

def check_api_key():
    """Check if Gemini API key is set."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("⚠️  Warning: GEMINI_API_KEY not found in environment variables.")
        print("\nTo set your API key:")
        print("  macOS/Linux: export GEMINI_API_KEY='your_api_key_here'")
        print("  Windows: set GEMINI_API_KEY=your_api_key_here")
        print("\nOr create a .env file with: GEMINI_API_KEY=your_api_key_here")
        return False
    else:
        print("✅ Gemini API key found!")
        return True

def main():
    """Main setup function."""
    print("🏥 Medical AI Demos Setup")
    print("=" * 40)
    
    # Check Python version
    check_python_version()
    
    # Install requirements
    install_requirements()
    
    # Check API key
    api_key_found = check_api_key()
    
    print("\n" + "=" * 40)
    print("🎉 Setup complete!")
    
    if api_key_found:
        print("\n🚀 Ready to run! Execute:")
        print("   streamlit run Hello.py")
    else:
        print("\n⚠️  Set your GEMINI_API_KEY first, then run:")
        print("   streamlit run Hello.py")
    
    print("\n📖 For more information, see README.md")

if __name__ == "__main__":
    main() 