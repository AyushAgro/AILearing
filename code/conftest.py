"""Put src/ on sys.path so `pytest` finds the forge package without installing it."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
