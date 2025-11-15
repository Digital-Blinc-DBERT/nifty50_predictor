#!/usr/bin/env python3
"""
EMERGENCY LAUNCHER for NIFTY 50 Predictor
Bypasses all pip/setuptools issues with Python 3.12
"""

import sys
import subprocess
import os

def emergency_install():
    """Emergency installation bypassing all build issues"""
    print("🚨 EMERGENCY LAUNCHER - Python 3.12 Compatibility Mode")
    print("=" * 60)

    # Force upgrade core tools first
    print("🔧 Step 1: Upgrading pip and setuptools...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "--upgrade", 
            "pip", "setuptools", "wheel", "--quiet"
        ], check=True)
        print("✅ Core tools upgraded")
    except:
        print("⚠️ Core tools upgrade failed, continuing...")

    # Install only pre-compiled packages to avoid build errors
    packages = [
        "streamlit",
        "pandas", 
        "numpy",
        "yfinance",
        "scikit-learn", 
        "plotly",
        "requests"
    ]

    print("📦 Step 2: Installing packages (pre-compiled only)...")
    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.run([
                sys.executable, "-m", "pip", "install", 
                "--only-binary=all", package, "--quiet", "--force-reinstall"
            ], check=True)
            print(f"✅ {package} installed")
        except subprocess.CalledProcessError:
            try:
                # Fallback method
                print(f"Retrying {package} with fallback method...")
                subprocess.run([
                    sys.executable, "-m", "pip", "install", 
                    package, "--user", "--quiet"
                ], check=True)
                print(f"✅ {package} installed (user)")
            except:
                print(f"❌ {package} failed - app may have limited functionality")

    # Test imports
    print("🧪 Step 3: Testing imports...")
    failed_imports = []

    test_imports = {
        "streamlit": "Streamlit web framework",
        "pandas": "Data processing",
        "yfinance": "Stock data fetching", 
        "sklearn": "Machine learning",
        "plotly": "Interactive charts",
        "numpy": "Numerical computing"
    }

    for module, description in test_imports.items():
        try:
            __import__(module)
            print(f"✅ {module} - {description}")
        except ImportError:
            failed_imports.append(module)
            print(f"❌ {module} - {description}")

    if failed_imports:
        print(f"\n⚠️ {len(failed_imports)} packages failed to import")
        print("App will run with reduced functionality")
    else:
        print("\n🎉 All packages imported successfully!")

    return len(failed_imports) == 0

def launch_app():
    """Launch the Streamlit app"""
    print("\n🚀 Step 4: Starting NIFTY 50 Predictor...")

    if not os.path.exists("app.py"):
        print("❌ app.py not found!")
        print("Make sure you're in the correct directory")
        input("Press Enter to exit...")
        return

    print("📊 Launching on http://localhost:8501")
    print("🌐 Browser should open automatically")
    print("⏹️ Press Ctrl+C here to stop the app")
    print("\n" + "=" * 60)

    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.headless", "true",
            "--server.port", "8501"
        ])
    except KeyboardInterrupt:
        print("\n🛑 App stopped by user")
    except FileNotFoundError:
        print("\n❌ Streamlit not found. Installation may have failed.")
        print("Try running: python -m pip install streamlit --force-reinstall")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTry manual launch: streamlit run app.py")

    input("\nPress Enter to exit...")

if __name__ == "__main__":
    print("Python version:", sys.version)

    # Run emergency installation
    install_success = emergency_install()

    if install_success:
        print("\n🎯 Installation successful! Launching app...")
    else:
        print("\n⚠️ Some packages failed, but trying to launch anyway...")

    # Launch the app
    launch_app()
