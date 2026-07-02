import os
import sys

# Make both the repo root (for `configs`) and the src/ layout
# (for `from models.models import ...`) importable.
_ROOT = os.path.dirname(__file__)
for _p in (_ROOT, os.path.join(_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
