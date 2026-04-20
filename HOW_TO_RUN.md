# HOW TO RUN CIPHER

This document covers everything you need to know to execute the CIPHER reinforcement learning environment natively on your machine structure.

## Prerequisites

- **Python 3.10+** (3.12 recommended)
- Dependencies: Install requirements via `pip install -r requirements.txt`. You will mainly need `networkx`, `pytest`, `numpy`, and `pydantic`.

## Directory Structure expectations

Ensure your code is located in the root repository folder alongside this document.

```
CIPHER/
├── cipher/
│   ├── agents/
│   ├── environment/
│   ├── memory/
│   ├── rewards/
│   ├── training/
│   └── utils/
├── tests/
├── main.py
├── README.md
└── HOW_TO_RUN.md
```

## Running an Episode

You can execute a single episode of CIPHER (which will spin up the environment, instantiate the agents, and run the main step loop) starting with `main.py`.

```bash
python main.py
```

`main.py` interfaces with `cipher.training.episode_runner.run_episode`. 

In standard behavior:
1. The 50-node graph generates.
2. The scenario and mission briefings generate based on previous win states.
3. RED and BLUE teams instantiate and collect observations.
4. The episode processes step-by-step up to the max step count (default: 50).
5. Output logs represent anomaly traces, agent movements, and reward computation.

## Running Tests

We implement strict validation to guarantee the asymmetry engines function exactly as intended. Execute the Phase 1 & 2 test suite directly:

```bash
pytest tests/test_env.py -v
```

This ensures the network topology bounds are respected, serialization roundtrips operate without loss, and asymmetric masking behaves correctly.

## Configuration

You can bypass default constants in `cipher/utils/config.py` by setting environment variables in your active shell:

- `CIPHER_ENV_GRAPH_SIZE=50`
- `CIPHER_ENV_MAX_STEPS=100`

See `config.py` for all available environment controls.
