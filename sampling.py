"""
Wrapper script for sampling scenes. This script uses custom_sample_and_render.py
with Hydra configuration.
"""
import sys
from pathlib import Path

# Add the scripts directory to the path
project_root = Path(__file__).parent
scripts_dir = project_root / "scripts"
sys.path.insert(0, str(scripts_dir))

# Import and run the main function from custom_sample_and_render
# This allows Hydra to work properly with the command-line arguments
if __name__ == "__main__":
    # Import here so Hydra can set up properly
    from custom_sample_and_render import main
    main()




