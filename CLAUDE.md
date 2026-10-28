# Notes for Claude sessions

## Machines
- CPU server: kenxu.top, repo at /home/main/yachiyo. VOICEVOX runs here:
  render clips that need a VOICEVOX voice (e.g. lafei) on this machine.
- GPU machine: WSL on Windows, RTX 3090 Ti 24 GB, repo at /mnt/e/Tsukuyomi,
  venv `tsukivenv`, run outputs in ~/tsk-runs. Training, evals and most
  renders run here. No VOICEVOX: Docker Hub is unreachable without the
  7890 proxy. Shell traffic goes through a proxy on 127.0.0.1:7890, so
  NO_PROXY must list 127.0.0.1,localhost.
- The two machines can't ssh to each other; move files with scp from
  PowerShell or through git.
