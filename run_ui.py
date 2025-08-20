#!/usr/bin/env python3
"""
Launcher script for the Mythologizer Textual UI application.
"""

import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mythologizer_core.ui import MythologizerApp

def main():
    """Launch the Mythologizer UI application."""
    try:
        app = MythologizerApp()
        app.run()
    except KeyboardInterrupt:
        print("\nApplication terminated by user.")
    except Exception as e:
        print(f"Error launching application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
