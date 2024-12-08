from functools import partial
import os
import sys


tqdm_mode = os.environ.get("FEEDBAX_TQDM", "auto")


match tqdm_mode:
    case "notebook":
        from tqdm.notebook import tqdm as _tqdm
    case "cli" | "console":
        from tqdm import tqdm as _tqdm
        _tqdm_write = partial(_tqdm.write, file=sys.stderr, end="")
    case "auto" | _:
        from tqdm.auto import tqdm as _tqdm
        _tqdm_write = partial(_tqdm.write, file=sys.stdout, end="")