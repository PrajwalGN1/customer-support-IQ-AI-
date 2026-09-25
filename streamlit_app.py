"""
Root Streamlit Entrypoint for Streamlit Community Cloud.
Allows deploying directly by pointing to either 'streamlit_app.py' or 'ui/streamlit_app.py'.
"""
import runpy
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Execute the main UI dashboard script
target_script = BASE_DIR / "ui" / "streamlit_app.py"
runpy.run_path(str(target_script), run_name="__main__")
