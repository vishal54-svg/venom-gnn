# Venom-GNN

Graph Neural Networks for identifying adversarial poisoning attacks in decentralized network topologies.

This is an academic project that uses **synthetic** network data. Results from this system are simulated experiments, not proof of real-world cybersecurity performance.

## Current status

**Phase 1 complete:** workspace inspected, project folder created, git initialized, directory skeleton in place.

Do not run training or the dashboard yet. Those arrive in later phases.

## Requirements (verified on this Windows machine)

| Tool | Status |
|------|--------|
| Git | Installed (`git version 2.56.0.windows.2`) |
| Node.js | Installed (`v24.19.0`, npm `11.17.0`) |
| Python | **Not usable** — `python.exe` is missing from leftover Python 3.12/3.13 folders; PATH currently hits MySQL Workbench Python |
| Ollama | Installed at `%LOCALAPPDATA%\Programs\Ollama\ollama.exe` (not on PATH); API at `http://localhost:11434` was not responding during Phase 1 |

## First command (required before Phase 2)

Install a working Python 3.11 or 3.12 from [python.org](https://www.python.org/downloads/), and tick **Add python.exe to PATH**.

Then in PowerShell:

```powershell
cd D:\venom-gnn
py -3 --version
```

If `py` still points at the broken 3.14 shim, use the full path printed by the installer, for example:

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" --version
```

## Layout

```
venom-gnn/
├── backend/          # FastAPI + ML (Phases 2–7)
├── frontend/         # React dashboard (Phase 8)
├── experiments/      # Experiment configs and run artifacts
├── results/          # Saved metrics from actual model runs
└── docs/             # Architecture, methodology, viva notes (Phase 10)
```
