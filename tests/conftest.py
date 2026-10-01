import os
import sys

# A tesztek a projektgyökérből importálnak (config, core).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
