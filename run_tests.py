import sys
import unittest
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run(verbosity: int = 2, pattern: str = "test*.py") -> bool:
    loader  = unittest.TestLoader()
    suite   = loader.discover(
        start_dir = str(PROJECT_ROOT / "tests"),
        pattern   = pattern,
    )

    runner  = unittest.TextTestRunner(verbosity=verbosity, stream=sys.stdout)
    result  = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    verbose = 2 if "-v" in sys.argv else 1

    # Simple -k filter: skip tests whose module name doesn't match
    pattern = "test*.py"
    for arg in sys.argv[1:]:
        if arg.startswith("-k"):
            keyword = arg[2:].strip() or (sys.argv[sys.argv.index(arg) + 1]
                                           if arg == "-k" else "")
            pattern = f"test*{keyword}*.py"
            break

    success = run(verbosity=verbose, pattern=pattern)
    sys.exit(0 if success else 1)
