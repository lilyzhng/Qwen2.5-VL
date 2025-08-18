#!/usr/bin/env python3
"""
Launcher script for NVIDIA Cosmos Video Retrieval System demonstrations.
"""

import sys
import subprocess
from pathlib import Path
import argparse


def run_unit_tests():
    """Run the unit tests for the visualizer."""
    print("🧪 Running Unit Tests...")
    print("-" * 40)
    try:
        result = subprocess.run([sys.executable, "test_visualizer.py"], 
                              capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"Failed to run tests: {e}")
        return False


def run_visualizer_demo():
    """Run the visualizer demonstration."""
    print("🎬 Running Visualizer Demo...")
    print("-" * 40)
    try:
        result = subprocess.run([sys.executable, "visualizer_demo.py"], 
                              capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"Failed to run demo: {e}")
        return False


def run_streamlit_app():
    """Launch the Streamlit web interface."""
    print("🌐 Launching Streamlit Web Interface...")
    print("-" * 40)
    print("This will open a web browser with the interactive interface.")
    print("Press Ctrl+C to stop the server.")
    print()
    
    try:
        # Check if streamlit is installed
        subprocess.run([sys.executable, "-c", "import streamlit"], 
                      check=True, capture_output=True)
        
        # Launch streamlit
        subprocess.run(["streamlit", "run", "streamlit_app.py"])
        
    except subprocess.CalledProcessError:
        print("❌ Streamlit not installed!")
        print("Install with: pip install streamlit")
        return False
    except Exception as e:
        print(f"Failed to launch Streamlit: {e}")
        return False


def run_benchmark():
    """Run performance benchmarks."""
    print("⚡ Running Performance Benchmarks...")
    print("-" * 40)
    
    # Check if video database exists
    video_db_path = Path("videos/video_database")
    if not video_db_path.exists():
        print("❌ Video database not found!")
        print(f"Expected path: {video_db_path.absolute()}")
        print("Please ensure you have video files in the database directory.")
        return False
    
    try:
        result = subprocess.run([sys.executable, "benchmark_optimizations.py"], 
                              capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"Failed to run benchmark: {e}")
        return False


def check_dependencies():
    """Check if required dependencies are installed."""
    print("🔍 Checking Dependencies...")
    print("-" * 40)
    
    required_packages = [
        "torch", "transformers", "numpy", "matplotlib", 
        "opencv-python", "pandas", "tqdm", "decord"
    ]
    
    optional_packages = [
        ("faiss-cpu", "FAISS for optimized search"),
        ("streamlit", "Web interface"),
        ("plotly", "Interactive visualizations"),
        ("pyarrow", "Parquet support")
    ]
    
    missing_required = []
    missing_optional = []
    
    # Check required packages
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_required.append(package)
    
    # Check optional packages
    for package, description in optional_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package} ({description})")
        except ImportError:
            print(f"⚠️  {package} ({description}) - optional")
            missing_optional.append(package)
    
    if missing_required:
        print(f"\n❌ Missing required packages: {', '.join(missing_required)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    if missing_optional:
        print(f"\n⚠️  Missing optional packages: {', '.join(missing_optional)}")
        print("Some features may not be available.")
    
    print("\n✅ All required dependencies found!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="NVIDIA Cosmos Video Retrieval System Demo Launcher"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Check command
    subparsers.add_parser('check', help='Check dependencies')
    
    # Test command
    subparsers.add_parser('test', help='Run unit tests')
    
    # Demo command
    subparsers.add_parser('demo', help='Run visualizer demonstration')
    
    # Web command
    subparsers.add_parser('web', help='Launch Streamlit web interface')
    
    # Benchmark command
    subparsers.add_parser('benchmark', help='Run performance benchmarks')
    
    # All command
    subparsers.add_parser('all', help='Run all demonstrations')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    print("🚀 NVIDIA Cosmos Video Retrieval System")
    print("=" * 50)
    
    if args.command == 'check':
        success = check_dependencies()
        sys.exit(0 if success else 1)
    
    elif args.command == 'test':
        success = run_unit_tests()
        sys.exit(0 if success else 1)
    
    elif args.command == 'demo':
        success = run_visualizer_demo()
        sys.exit(0 if success else 1)
    
    elif args.command == 'web':
        run_streamlit_app()
    
    elif args.command == 'benchmark':
        success = run_benchmark()
        sys.exit(0 if success else 1)
    
    elif args.command == 'all':
        print("Running complete demonstration suite...\n")
        
        # Check dependencies first
        if not check_dependencies():
            print("\n❌ Dependency check failed!")
            sys.exit(1)
        
        # Run tests
        print("\n" + "="*50)
        if not run_unit_tests():
            print("\n⚠️  Unit tests failed, continuing anyway...")
        
        # Run demo
        print("\n" + "="*50)
        if not run_visualizer_demo():
            print("\n⚠️  Demo failed, continuing anyway...")
        
        # Run benchmark if possible
        print("\n" + "="*50)
        if not run_benchmark():
            print("\n⚠️  Benchmark failed (may need video files)")
        
        print("\n" + "="*50)
        print("🎉 Demonstration suite completed!")
        print("\nTo try the interactive web interface:")
        print("  python run_demo.py web")


if __name__ == "__main__":
    main()
