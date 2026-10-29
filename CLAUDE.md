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

## Avatars and rights
- 八千代辉夜姬 (fan-made, watermarked) and the Azur Lane rigs
  (huangjiafangzhou_3 etc.): train and test only; never stream with them,
  never commit their model files. Mao and the other Cubism samples are
  under Live2D's free material licence.
- r2 avatars planned: Hiyori, lafei (limited), Mao, huangjiafangzhou_3
  (capable), 八千代辉夜姬 (face-first, no arm controls), seed-san (VRM).
