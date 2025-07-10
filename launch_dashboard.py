#!/usr/bin/env python3
"""
Launch script for LLMTraceFX Real-Time Dashboard
"""

import subprocess
import sys
import os
import logging
from datetime import datetime

# Configure logging for launch script
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('llmtracefx_launcher.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Launch the Streamlit dashboard"""
    logger.info("Starting LLMTraceFX Dashboard Launcher")
    
    dashboard_path = os.path.join("llmtracefx", "realtime_dashboard.py")
    
    # Check if dashboard file exists
    if not os.path.exists(dashboard_path):
        logger.error(f"Dashboard file not found: {dashboard_path}")
        print(f"Error: Dashboard file not found: {dashboard_path}")
        sys.exit(1)
    
    logger.info(f"Dashboard path: {dashboard_path}")
    logger.info("Launching Streamlit server...")
    
    try:
        # Launch Streamlit dashboard
        cmd = [
            sys.executable, 
            "-m", "streamlit", 
            "run", 
            dashboard_path,
            "--server.port=8501",
            "--server.address=0.0.0.0",
            "--server.headless=true",
            "--browser.gatherUsageStats=false"
        ]
        
        logger.info(f"Executing command: {' '.join(cmd)}")
        print("🚀 Starting LLMTraceFX Real-Time Dashboard...")
        print("📊 Dashboard will be available at: http://0.0.0.0:8501")
        print("📝 Logs will be written to: llmtracefx_dashboard.log")
        print("🔄 Press Ctrl+C to stop the dashboard")
        
        subprocess.run(cmd, check=True)
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error launching dashboard: {e}")
        print(f"Error launching dashboard: {e}")
        sys.exit(1)
    except FileNotFoundError:
        logger.error("Streamlit not found in Python environment")
        print("Error: Streamlit not found. Please install it with: pip install streamlit")
        print("Or run: uv sync  # to install all dependencies")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Dashboard stopped by user")
        print("\n🛑 Dashboard stopped")
        sys.exit(0)

if __name__ == "__main__":
    main()
