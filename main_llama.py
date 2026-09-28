import os
import sys

from orion.app import main


if __name__ == "__main__":
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        os.environ.setdefault("ORION_MODEL_PATH", sys.argv.pop(1))
        os.environ.setdefault("ORION_MODEL_BACKEND", "llama.cpp")
    raise SystemExit(main())
