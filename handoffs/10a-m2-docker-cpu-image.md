# Branch 10a Handoff: CPU-Only Docker Image

## Why this branch exists

Branch `10a-m2-docker-cpu-image` starts from `10-m2-containerization` (commit
`c127ecc`) and fixes how the branch-10 image is built, so the fix lives in the
containerization phase it belongs to. It does not change any API, retrieval,
or generation code. The same Dockerfile change was first built and tested as
`12a-m2-docker-cpu-image` on top of branch 12 (commit `2cb2c26`); branch 10a
repeats that test on branch 10's own code.

Branches 11 and 12 are cumulative and do not get this fix automatically. After
this branch is merged into `10-m2-containerization`, merge 10 into 11 and 11
into 12 (no conflicts were found when simulated with `git merge-tree`).

Building the branch-10 Dockerfile unchanged showed:

| Finding | Evidence |
|---|---|
| pip installed the CUDA build of PyTorch | `torch 2.15.0+cu132`, `torch.version.cuda == "13.2"` |
| CUDA libraries the container can never use | `site-packages/nvidia` = 3.0 GB, `site-packages/torch` = 1.2 GB |
| Image size | 10.3 GB |
| Models fetched at every fresh container start | `all-MiniLM-L6-v2` and `cross-encoder/ms-marco-MiniLM-L-6-v2` downloaded from huggingface.co during FastAPI startup |
| Branch-10 HANDOUT test command | sends `{"question": ...}`; the API field is `query` and extra fields return HTTP 422; `curl.exe -d '{...}'` also loses its inner quotes in Windows PowerShell 5.1 |

## Changes

- `Dockerfile`
  - installs `torch==2.15.0` from `https://download.pytorch.org/whl/cpu`
    before `requirements-api.txt`, then fails the build if a CUDA torch is
    present;
  - downloads both Hugging Face models at build time into `HF_HOME=/opt/hf-cache`
    as the `app` user, in a layer before `COPY src` so code edits do not
    re-download them;
  - sets `HF_HUB_OFFLINE=1` so the running container never calls huggingface.co;
  - `--build-arg PRELOAD_MODELS=0` restores download-on-start for offline builds;
  - port, `PORT`, healthcheck, one worker, non-root user, and the two BM25
    artifacts are unchanged.
- `compose.yaml`: `docker compose up --build` with `.env`, port 8000, and a
  2 GB memory limit that mirrors a 2 GiB Cloud Run or paid Render instance.
- `tests/test_container_config.py`: asserts the CPU index and CUDA guard, and
  that the preloaded model names equal `EMBEDDING_MODEL` and `RERANKER_MODEL`
  in `src/retrieval.py` (read with `ast`, so the test stays lightweight).
- `README.md`: compose usage, `.env` format traps, PowerShell request example,
  memory requirement, and `--platform linux/amd64` for Apple Silicon builds.
- `HANDOUT.md`: request examples use `query`, `/predict` smoke tests use
  `Invoke-RestMethod`, and the Compose alternative and new files are listed.

## Verification status

| Check | Where | Result |
|---|---|---|
| Ruff | local | passed |
| `tests/test_container_config.py` | local | 5 passed |
| `docker build --check` | local | no warnings |
| Full pytest, Docker build with CPU torch and model preload | GitHub Actions on push | see the run for this branch |
| `/health`, `/docs`, `/predict` for both strategies | Windows laptop with real keys | to do, steps below |
| Container memory with both retrievers loaded | `docker stats` on the laptop | to do, record the number |

## Run it on Windows (PowerShell)

Prerequisite: Docker Desktop running with the WSL 2 engine. `docker version`
must show both a Client and a Server section.

```powershell
git clone https://github.com/<your-account>/MLOps-for-Legal-Aid-RAG-pipeline.git
cd MLOps-for-Legal-Aid-RAG-pipeline
git switch 10a-m2-docker-cpu-image
notepad .env
```

In Notepad, write exactly two lines with no quotes and save (Notepad saves
UTF-8; `echo ... > .env` in Windows PowerShell 5.1 saves UTF-16, which Docker
rejects):

```text
PINECONE_API_KEY=your-pinecone-key
GROQ_API_KEY=your-groq-key
```

Build and start (the first build downloads about 1 GB and can take several
minutes; later builds reuse cached layers):

```powershell
docker compose up --build --detach
docker compose logs --follow api
```

Wait for `Application startup complete`, press Ctrl+C to stop following logs,
then check:

```powershell
docker compose ps
Invoke-RestMethod http://127.0.0.1:8000/health
$body = @{ query = "Who inherits property when a person dies without a will?"; chunking_strategy = "recursive" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -ContentType "application/json" -Body $body
$body = @{ query = "Who inherits property when a person dies without a will?"; chunking_strategy = "fixed" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -ContentType "application/json" -Body $body
docker stats --no-stream legal-aid-rag-api
```

Expected:

- `docker compose ps` shows `(healthy)`;
- `/health` returns `status: ok`, `retriever_loaded: True`, both strategies;
- each `/predict` returns an `answer`, `sources`, and `timings`;
- `http://127.0.0.1:8000/docs` opens in the browser;
- record the `MEM USAGE` from `docker stats` in `MLOPS_LOG.md`. It decides
  whether any 512 MB host is possible.

Stop and clean up:

```powershell
docker compose down
```

## If Docker Desktop will not start on Windows

The error seen on the original laptop,
`Wsl/Service/RegisterDistro/CreateVm/HCS/ERROR_FILE_NOT_FOUND`, comes from
WSL/Hyper-V, not from this Dockerfile. Try in order, from an Administrator
PowerShell:

1. `wsl --update`, then `wsl --shutdown`, then restart Docker Desktop.
2. Turn on the Windows features WSL needs, then reboot:
   `dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart`
   and
   `dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart`.
3. Task Manager > Performance > CPU must show `Virtualization: Enabled`;
   otherwise enable it (Intel VT-x / AMD SVM) in the BIOS/UEFI settings.
4. `wsl --list --verbose` should list `docker-desktop`. If it is missing or
   broken, the remaining fixes are Docker Desktop > Troubleshoot > Reset to
   factory defaults, or `wsl --unregister docker-desktop`. Both delete every
   local Docker image, container, and volume, so agree on it first.

## Merging

Merge this branch into `10-m2-containerization`, then carry it forward
(10 into 11, 11 into 12). None of these branches deploy. Only `main` deploys:
branch 12's `render.yaml` sets `autoDeployTrigger: checksPass` on `main` with
the free plan, which reproduces the 512 MiB out-of-memory failure. Pause
Render auto-deploy, or move to a host with about 2 GiB, before this fix
reaches `main`.
