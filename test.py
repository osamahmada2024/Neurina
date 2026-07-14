from pathlib import Path


print("Current directory:", Path(__file__).resolve())

import gfpgan
import sys

print(sys.modules.get("gfpgan"))
import sys
print(sys.modules.get("gfpgan"))