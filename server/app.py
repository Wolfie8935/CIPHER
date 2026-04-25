"""
FastAPI app for CIPHER via OpenEnv Core (multi-mode deployment / `openenv validate`).

Run:
  uv run --project . server
  uv run --project . server --port 8000
  uvicorn server.app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import sys
from pathlib import Path

# Repo root on PYTHONPATH so `cipher` and `server` resolve for `uv run` and local runs
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from openenv_core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv_core is required. Install with: uv sync\n"
    ) from e

from server.cipher_http_env import CIPHERAction, CIPHERHTTPEnvironment, CIPHERObservation

_env = CIPHERHTTPEnvironment()
app = create_app(_env, CIPHERAction, CIPHERObservation, env_name="cipher")


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    # Entrypoint can also be run via: python -c "import server.app as m; m.main()"
    # `main()` (script hook): pyproject [project.scripts] server = server.app:main
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
