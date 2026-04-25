"""
OpenEnv ``openenv validate`` expects ``server/app.py`` with ``main()`` and a
``__main__`` guard. This mirrors ``hf_app.py`` (Dash dashboard on PORT, default 7860).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> None:
    port = int(os.environ.get("PORT", "7860"))
    from cipher.dashboard.app import app

    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    main()
