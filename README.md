# CIPHER 
### Adversarial Multi-Agent Reinforcement Learning Environment

CIPHER is a production-grade, asymmetric multi-agent reinforcement learning environment designed for researching advanced adversarial behaviors in enterprise networks. It pitches an autonomous RED team against an autonomous BLUE team in a 50-node, 4-zone simulated corporate environment.

## Key Features

1. **Enterprise Network Simulation (Phase 2)**
   - 50-node topology spanning 4 distinct zones (DMZ, Corporate, Restricted, Critical).
   - Realistic hostnames, services, operating systems, and zone-aware firewall rules.
   - High-Value Targets (HVTs) residing deep within the Critical zone.

2. **Strict Information Asymmetry**
   - **RED Team**: Operates in a fog-of-war. Must discover the network dynamically. Sees hostnames, services, and local connections. Cannot distinguish honeypots from legitimate file servers.
   - **BLUE Team**: Operates via sensors. Cannot see RED directly. Tracks anomalies, responds to statistical exhaust, manages honeypots, and investigates suspicious nodes based on severity.

3. **Auto-Escalating Scenarios**
   - Difficulty automatically scales based on team performance (win streaks).
   - Elevated difficulty increases BLUE honeypot density and zone lockdown procedures.
   - Generates natural-language mission and defense briefings for respective teams.

4. **Advanced State Mechanics**
   - Privilege tracking and credential harvesting requirements.
   - Distinct anomaly classifications (e.g., unusual traversal, lateral movement bursts, privilege escalation attempts).
   - MEMENTO Memory System / Dead Drop vaults allowing agents to pass information across temporal context boundaries.

## Architecture

CIPHER is structured around a centralized episode runner with deep decoupling between ground-truth state, observations, and agent logic. 

- `cipher.environment.state`: Ground truth tracking (movement, suspicion, anomalies).
- `cipher.environment.observation`: The asymmetry engine enforcing fog-of-war.
- `cipher.environment.graph`: Network topology generator.
- `cipher.environment.scenario`: Dynamic difficulty adjustment engine.
- `cipher.training._episode_runner`: The main event loop synchronizing steps.

## Status

**Phases 1 & 2** are complete. 
- The project skeleton is robust and production-ready.
- The 50-node enterprise environment with asymmetric observations and an active test suite (47 tests) validates all core mechanics.

## Testing

Ensure you have your environment configured properly (see `HOW_TO_RUN.md`). To run the 47-test validation suite:

```bash
pytest tests/test_env.py
```
