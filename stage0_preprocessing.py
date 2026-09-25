"""
Forwarding stub for Stage 0 Preprocessing.
Canonical implementation is located in:
code/business_entity_resolution/src/stage0_preprocessing.py
"""
import sys
import os

# Add code/business_entity_resolution to sys.path
_code_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "code", "business_entity_resolution"))
if _code_dir not in sys.path:
    sys.path.insert(0, _code_dir)

from src.stage0_preprocessing import *

if __name__ == "__main__":
    # Execute the self-test from canonical module
    import subprocess
    target = os.path.join(_code_dir, "src", "stage0_preprocessing.py")
    sys.exit(subprocess.call([sys.executable, target]))
