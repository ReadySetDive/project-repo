---
tags: [infra, carc, hpc, hypersteer]
aliases: [CARC, Discovery, USC HPC]
status: in-progress
updated: 2026-10-06
---

# USC CARC (Discovery) Wiki

How to use USC's CARC cluster (Discovery) for HyperSteer training instead of RunPod. Everything here was verified on 2026-10-06 unless marked *(unverified)*.

---

## 1. Quick reference

| Thing | Value |
|---|---|
| Login host | `discovery.usc.edu` (login nodes `10.72.0.13`, `10.72.0.14`) |
| SSH alias (local) | `ssh carc` (see [§3](#3-ssh-key-setup-one-time)) |
| Username | `mgordy` (USC NetID) |
| Project / Slurm account | `xiangren_1987` (default account; likely the class allocation shared with classmates) |
| SU budget | 600,000 limit; usage 33,342 as of 2026-10-07 (decays, half-life 7 d; see [§7](#7-cost-service-units)) |
| Allowed partitions | `debug`, `main`, `gpu`, `oneweek`, `largemem`, `htcondor` |
| Network requirement | USC VPN off campus (Cisco Secure Client / AnyConnect, `vpn.usc.edu` + Duo) |

---

## 2. Connecting

1. **Off campus → connect the USC VPN first.** Without it, `discovery.usc.edu` does not even resolve:
   ```
   ssh: Could not resolve hostname discovery.usc.edu: No such host is known.
   ```
   On campus, USC Secure Wireless or wired works; guest Wi-Fi does not.
2. `ssh carc` (after key setup) or `ssh mgordy@discovery.usc.edu` (password + Duo).
3. If DNS fails while on the VPN, use the IP directly: `ssh mgordy@10.72.0.13`.

> Submitted jobs keep running when you disconnect / drop the VPN. The VPN is only needed to log in, submit, check, or move files.

---

## 3. SSH key setup (one-time)

Done 2026-10-06. Recorded so it can be redone on another machine.

**Local (Windows):**
- Key: `~/.ssh/id_ed25519_carc` (+ `.pub`), dedicated to CARC, no passphrase.
- `~/.ssh/config` entry:
  ```
  Host carc
    HostName discovery.usc.edu
    User mgordy
    IdentityFile ~/.ssh/id_ed25519_carc
    IdentitiesOnly yes
    ServerAliveInterval 60
  ```

**On CARC** (logged in with password), append the public key:
```bash
mkdir -p ~/.ssh && chmod 700 ~/.ssh
echo 'ssh-ed25519 AAAA... mgordy@carc' >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

**Gotchas hit along the way:**
- `.ssh` lives in your home directory (`/home1/mgordy/.ssh`), same as `C:\Users\<you>\.ssh` on Windows.
- Plain `ls` hides dotfiles → use `ls -a`. `mkdir -p` prints **nothing** on success (and nothing if the dir already exists).
- `ssh mgordy@discovery.usc.edu` does **not** use the new key (non-default filename) → use `ssh carc` or `ssh -i ~/.ssh/id_ed25519_carc ...`.
- Duplicate lines in `authorized_keys` are harmless. If `sort -u` doesn't dedupe them, they differ invisibly (trailing space / `\r`); check with `cat -A`.
- **Never** use `> file` to rewrite a file you're reading from; `>>` appends, `sort -u f -o f` is safe.

---

## 4. Running commands non-interactively

CARC's helper commands (`myaccount`, `noderes`, `jobinfo`) are only on `PATH` in a **login shell**:

```powershell
ssh carc "myaccount"                 # -> bash: myaccount: command not found
ssh carc "bash -lc 'myaccount'"      # works
```

Interactively (after `ssh carc`) just type them.

| Command | What it shows |
|---|---|
| `myaccount` | Accounts, SU limit/usage/remaining, allowed partitions |
| `noderes -f -g` | GPU nodes with **free** GPUs right now (only free ones!) |
| `noderes -c -g` | **All** configured GPU nodes |
| `squeue --me` | Your queued/running jobs |
| `jobinfo <jobid>` | Details (incl. resource use) of a finished job |
| `sinfo -p gpu -N -o "%N %G %f"` | Per-node GRES + features (e.g. `a100-80gb`) *(not yet run)* |

---

## 5. GPU inventory (Discovery)

From `noderes -c -g`, 2026-10-06:

| Partition | GPU | VRAM | Nodes | GPUs | Max/node |
|---|---|---|---|---|---|
| gpu | **A100** | 40 or 80 GB | 20 | **40** | 2 |
| gpu | A40 | 48 GB | 27 | 54 | 2 |
| gpu | L40S | 48 GB | 7 | 22 | 3 |
| gpu | V100 | 32 GB | 27 | 54 | 2 |
| gpu | P100 | 16 GB | 38 | 76 | 2 |
| debug | A40 | 48 GB | 1 | 2 | 2 |
| debug | P100 | 16 GB | 1 | 2 | 2 |
| **Total** | | | **121** | **250** | |

- **A100 split (from `sinfo`, 2026-10-07): 16× A100-80GB on 8 nodes, 24× A100-40GB on 12 nodes.** Request 80 GB with `#SBATCH --constraint=a100-80gb`.
- `htcondor`, `main`, `oneweek`, `largemem` have **no GPUs**.
- **Live snapshot script:** `code/hypersteer/scripts/carc_gpus.ps1` (read-only; `-Raw` dumps raw lines + pending reasons). Per type: how many **nodes** could start a 1/2/3-card job right now (a multi-card job must fit on one node, and the node also needs CPUs + RAM free: `-MinCpus 8 -MinRamGB 64`), plus queue demand and a per-node list of free cards/CPUs/RAM. Snapshot 2026-10-07 06:23: 53 of 250 cards free, **0 A100-80GB free, ~128 GPUs' worth of jobs queued for those 16 cards**; A40/L40S queues near empty.
- Recount: `sinfo -N -h -o "%N|%P|%G|%f"` (GRES column = `gpu:<type>:<n>`, features column has `a100-80gb`/`a100-40gb`).
- A100/A40 nodes were mostly `mixed`/`allocated`/`plnd` (planned for queued jobs). **Expect queue waits for A100s.**
- Per-user caps on `gpu` (QOS `gpu`): **12× A100, 12× A40, 6× L40S, 36 GPUs total, 400 CPUs** at once (`sacctmgr show qos gpu`).
- Node specs: A100 nodes = EPYC 7513, 64 CPUs, 248 GB RAM. A40 nodes = EPYC 7282, 32 CPUs, 248 GB RAM. L40S nodes = EPYC 9534, 64 CPUs, ~748 GB RAM *(L40S spec from search summary, unverified)*.

---

## 6. Partitions & time limits

| Partition | Max runtime | Use for |
|---|---|---|
| `debug` | 1 h | Smoke tests (A40, P100 available) |
| `gpu` | 48 h | Real training |
| `main`, `epyc-64` | 48 h | CPU jobs |
| `oneweek`, `largemem` | 168 h | Long CPU / big-memory jobs |

48 h cap on `gpu` → long runs must **checkpoint + resume** across chained jobs (`runpod_train.sh` already supports `RESUME=1`):
```bash
sbatch --dependency=afterany:<prev_jobid> carc_large.sbatch
```

---

## 7. Cost (Service Units)

*Verified 2026-10-07 against CARC docs, the live Slurm config, and real jobs on our account.*

### 7.1 The formula

You're charged **SUs per minute** for the resources you **reserve**, for as long as the job **actually runs**:

$$\text{SU} = \Big(\underbrace{1\cdot\text{CPUs}}_{\text{cores}} + \underbrace{0.25\cdot\text{RAM}_\text{GB}}_{\text{1 SU per 4 GB}} + 8\cdot N_{\text{A100/A40/L40S}} + 4\cdot N_{\text{V100/P100}}\Big) \times \text{elapsed minutes}$$

| Resource reserved for 1 min | SU |
|---|---|
| 1 CPU core | 1 |
| 1 GB RAM (system RAM, `--mem`) | 0.25 |
| 1× A100 (40 *or* 80 GB), A40, L40S | 8 |
| 1× V100, P100 | 4 |
| GPU VRAM | 0 (comes with the card) |

How we know each piece:
- **Rates:** CARC [Running jobs](https://www.carc.usc.edu/user-guides/hpc-systems/using-our-hpc-systems/running-jobs) table, *and* Slurm's own config on every partition: `TRESBillingWeights=CPU=1.0,Mem=0.25G,GRES/gpu:{a100,a40,l40s}=8.0,GRES/gpu:{v100,p100}=4.0` (`scontrol show partition gpu`). No `PriorityFlags=MAX_TRES`, so components are **summed** ([Slurm TRES docs](https://slurm.schedmd.com/tres.html)).
- **Reserved, not used:** CARC: *"SUs are charged based on resources reserved for your job, not the resources that your job actually uses."* Ask for 8 CPUs, use 1 → pay for 8. Same for RAM.
- **Elapsed time, not `--time`:** summing `billing × elapsed` over all 39 jobs on our account gives 34,460 SU vs `myaccount`'s 33,342 (the small gap is decay, §7.3). Billing on time limits would be far higher (e.g. a job that ran 104 min of a 240 min limit). Finishing early saves SUs.
- **Checked on real jobs:** a classmate's jobs (8 CPU, 64 GB, 1× L40S) bill `billing=32` = 8 + 16 + 8. ✔
- **Rounding:** Slurm stores each job's per-minute billing as an integer, **truncated** ([TRES docs](https://slurm.schedmd.com/tres.html)). E.g. 8 CPU + 50 GB + 1 A100 = 28.5 → 28.
- **Default RAM:** with no `--mem`, you get `DefMemPerCPU=2048` (2 GB per CPU = +0.5 SU/min per CPU).
- **A100-80GB costs the same as A100-40GB** (both `gpu:a100`).

### 7.2 Cost per job

| Job | Resources | SU/min | SU/hour | SU/24 h | % of 600k per 24 h |
|---|---|---|---|---|---|
| Smoke test (`debug`) | 1× A40, 4 CPU, 16 GB | 16 | 960 | (≤1 h max: 960) | 0.16% / h |
| 2B training | 1× A100, 8 CPU, 48 GB | 28 | 1,680 | 40,320 | 6.7% |
| 2B training, lean | 1× A100, 4 CPU, 16 GB | 16 | 960 | 23,040 | 3.8% |
| Classmate's actual jobs | 1× L40S, 8 CPU, 64 GB | 32 | 1,920 | 46,080 | 7.7% |
| 30B, 1 card (8-bit base) | 1× A100-80GB, 8 CPU, 64 GB | 32 | 1,920 | 46,080 | 7.7% |
| 30B, 2 cards | 2× A100, 16 CPU, 128 GB | 64 | 3,840 | 92,160 | 15.4% |
| 30B, 3 cards | 3× L40S, 24 CPU, 192 GB | 96 | 5,760 | 138,240 | 23.0% |
| Old cards | 1× V100, 8 CPU, 48 GB | 24 | 1,440 | 34,560 | 5.8% |

Where the SUs go: in a typical 1-GPU job the GPU is only 8 of ~28 SU/min. **CPUs and RAM are most of the bill**, so trim them to what the job measurably needs (`sacct -j <id> -o MaxRSS` after a test run; leave headroom, since Slurm kills jobs that exceed `--mem`).

Rough $ equivalence (for intuition only): RunPod A100 ≈ $1.05/h (our 19 h ≈ $20) vs ~1,680 SU/h here → **1,000 SU ≈ $0.60**, whole 600k ≈ **$375** of RunPod time.

### 7.3 Our budget and how the limit actually works

**Dollar cost to us: $0.** Discovery compute is free ([allocation types](https://www.carc.usc.edu/user-guides/project-and-allocation-management/allocation-types-and-requests)); the SU budget is the only limit.

| | |
|---|---|
| Account | `xiangren_1987`: Slurm `GrpTRESMins=billing=600000` (1 SU = 1 billing-minute) |
| Usage (2026-10-07) | **33,342 SU** (up from 7,182 on 10-06; one classmate running 1× L40S jobs) |
| Members | 283 users; 39 jobs total since Aug 2025, all in Oct 2026 |
| Tier | CARC's "Class" allocation is exactly 600,000 SU → almost certainly the CSCI 544 class pool |

**Usage decays (budget slowly refills).** `myaccount` reports Slurm's `GrpTRESRaw billing` (read from `/apps/utilities/myaccount`). Per [Slurm resource limits](https://slurm.schedmd.com/resource_limits.html), usage counted against `GrpTRESMins` *"is decayed (at a rate of PriorityDecayHalfLife)"*, and CARC sets `PriorityDecayHalfLife=7-00:00:00`, `PriorityUsageResetPeriod=NONE`, with no `NoDecay` flag on any QOS. So:
- Each SU you spend counts against the limit at full weight, then **half after 7 days, a quarter after 14**, …
- Sustainable burn rate for the whole account ≈ 600,000 / (7 d / ln 2) ≈ **59,000 SU/day ≈ 2,500 SU/h**, i.e. about **1.5 single-A100 jobs running 24/7 indefinitely** (for everyone combined).
- *Caveat:* this is Slurm's mechanics on CARC's config; CARC's docs don't mention decay, and staff could also reset/expire allocations administratively (allocations "expire either by date or when a limit is reached"). Don't plan around the refill without asking.

**What happens near the limit.** CARC runs `AccountingStorageEnforce=associations,limits,qos,safe`. With `safe`, *"a job will only be launched … if the job will be able to run to completion"* ([slurm.conf](https://slurm.schedmd.com/slurm.conf.html)):
- Running jobs are **not** killed when the pool runs low.
- But a job only **starts** if `remaining ≥ billing × --time`. Its `--time` matters for *admission* even though you're billed on elapsed time. A 48 h job at 28 SU/min needs 80,640 SU free to start; at 96 SU/min (3× L40S) it needs 276,480.
- Practical: set `--time` close to the real runtime (+ margin). Over-asking doesn't cost SUs but can block your start when the pool is low (and delays scheduling generally).

**Who's using it / check anytime:**
```bash
myaccount                                   # limit / usage / remaining (decayed)
sshare -A xiangren_1987 --all -o Account,User,RawUsage,GrpTRESRaw   # per user
sacctmgr -n -P show assoc account=xiangren_1987 format=User | sort -u | wc -l
jobinfo <jobid>                             # one finished job, incl. efficiency
```

**Bigger pools (free):** Graduate student project (GSP) = 6,000,000 SU (10× ours; a PI requests it in the CARC user portal); research Small = 12,000,000. Ask the instructor/advisor before any 30B campaign.

---

## 8. Storage

| Path | Quota | Backed up | Use for |
|---|---|---|---|
| `/home1/mgordy` | 100 GB, 1.91M files | Yes (partial snapshots, 2 wk) | Code, small configs, `.env` |
| `/project2/xiangren_1987` *(path inferred from account name, unverified)* | 5 TB default (shared with group) | No (2 wk partial snapshots) | Checkpoints you want to keep |
| `/scratch1/mgordy` | 10 TB | **No**, purged when FS > 80% full | HF cache, datasets, live checkpoints |

Set `HF_HOME=/scratch1/mgordy/hf` so model downloads don't blow the home quota. Copy finished checkpoints off scratch (to `/project2` or HF).

---

## 9. Moving code & data

- **Transfer nodes:** `hpc-transfer1.usc.edu`, `hpc-transfer2.usc.edu` (use these for big `rsync`s, not the login node). Globus endpoints too.
- **Preferred:** push repo to GitHub, `git clone` / `git pull` on CARC.
- **Or rsync** (Git Bash / WSL; plain PowerShell has no `rsync`):
  ```bash
  rsync -av \
    --exclude '.venv*' --exclude 'demo/.venv-win' --exclude 'assets/checkpoints' \
    --exclude '.gradio' --exclude '.env' --exclude 'logs' \
    code/hypersteer/ mgordy@hpc-transfer1.usc.edu:/home1/mgordy/hypersteer/
  ```
- **Secrets:** don't copy `.env`. Recreate it on CARC (`chmod 600 .env`) with a **separate** HF token.
- Compute-node internet access: *(unverified)*. Safest: pre-download model + dataset on the login node, then run jobs with `HF_HUB_OFFLINE=1`.

---

## 10. Job script template

```bash
#!/bin/bash
#SBATCH --account=xiangren_1987
#SBATCH --partition=gpu          # debug for <=1h smoke tests
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=a100:1   # or a40:1 / l40s:1
#SBATCH --constraint=a100-80gb   # optional, A100 only
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out

export HF_HOME=/scratch1/mgordy/hf
export HF_HUB_OFFLINE=1
bash scripts/carc_train.sh large
```

`scripts/runpod_train.sh` **cannot be used as-is**: it calls `runpodctl` and refuses to start when `AUTO_STOP=1` can't stop a pod. Needs a CARC copy (`carc_train.sh`) without auto-stop. *Not yet written.*

Workflow: `sbatch job.sbatch` → `squeue --me` → `tail -f logs/<name>_<jobid>.out` → `jobinfo <jobid>`.

---

## 10b. Scheduling jobs (Slurm)

**Submit**
| Command | What it does |
|---|---|
| `sbatch job.sbatch` | Queue a batch job; prints `Submitted batch job <jobid>` |
| `sbatch --begin=22:00 job.sbatch` | Don't start before a time (also `now+2hours`, `2026-10-07T09:00`) |
| `sbatch --dependency=afterok:<id> job.sbatch` | Start only after job `<id>` succeeds (`afterany` = regardless of exit) |
| `sbatch --array=0-9 job.sbatch` | Job array: 10 copies, index in `$SLURM_ARRAY_TASK_ID` |
| `salloc --partition=debug --gpus-per-task=a40:1 --time=1:00:00 --account=xiangren_1987` | Interactive GPU shell on a compute node; `exit` to release |

Email on start/end/fail: add `#SBATCH --mail-type=all` and `#SBATCH --mail-user=<email>`.
Output defaults to `slurm-<jobid>.out` in the submit directory; override with `--output`.

**Monitor / manage**
| Command | What it does |
|---|---|
| `squeue --me` | Your jobs. States: `PD` pending, `R` running, `CG` completing, `CD` done, `CA` cancelled |
| `squeue --me --start` | Slurm's **estimated start time** for pending jobs |
| `sprio -j <id>` | Priority breakdown of a pending job |
| `sstat -j <id> -o JobID,MaxRSS` | Live memory use of a running job |
| `sacct -j <id> -o JobID,State,ExitCode,Elapsed,MaxRSS` | Accounting after it ends |
| `jobinfo <id>` | CARC summary incl. efficiency (check SU charge here) |
| `scancel <id>` / `scancel --me` | Cancel one / all of your jobs |
| `scontrol hold <id>` / `scontrol release <id>` | Pause / unpause a **pending** job |
| `scontrol update JobId=<id> TimeLimit=12:00:00` | Change a pending job (can only lower time once running) |

**Per-user limits on Discovery** ([running jobs](https://www.carc.usc.edu/user-guides/hpc-systems/using-our-hpc-systems/running-jobs)):
| Partition | Max time | Max concurrent CPUs | Max concurrent GPUs | Max running jobs | Max queued jobs |
|---|---|---|---|---|---|
| `gpu` | 48 h | 400 | 36 | 36 | 100 |
| `debug` | 1 h | 48 | 4 | 5 | 5 |
| `main` + `epyc-64` | 48 h | 2,000 | 36 | 100 | 5,000 |
| `oneweek` | 168 h | 208 | n/a | 50 | 50 |
| `largemem` | 168 h | 64 | n/a | 3 | 10 |

Login nodes: 4 cores, 32 GB, 64 processes per user. **Don't train on the login node**; only edit, download, submit.

**Priority:** mostly **fairshare** = how much you *and your project* used in the last 30 days, plus how long the job has waited and its size. Since `xiangren_1987` is shared, classmates' usage lowers your priority too. Smaller requests (fewer GPUs/CPUs/RAM, shorter `--time`) start sooner, since they fit in backfill gaps. A pending reason of `Priority` just means others are ahead; `Resources` means waiting for hardware.

**Chaining long runs past 48 h:**
```bash
J1=$(sbatch --parsable carc_large.sbatch)
J2=$(sbatch --parsable --dependency=afterany:$J1 --export=ALL,RESUME=1 carc_large.sbatch)
J3=$(sbatch --parsable --dependency=afterany:$J2 --export=ALL,RESUME=1 carc_large.sbatch)
```
*(`--begin`, `--array`, `--dependency`, `hold`/`release` are standard Slurm from the [CARC cheatsheet](https://www.carc.usc.edu/user-guides/hpc-systems/using-our-hpc-systems/slurm-cheatsheet); the `--export`/`--parsable` chaining pattern is standard Slurm, not yet tested on CARC.)*

**Racing submissions (first to start cancels the rest).** When you'd take whichever GPU frees up first but the job's settings depend on the card (e.g. a token budget sized to what that GPU finishes before `--time` runs out), submit one copy per GPU type under the same `--job-name`, each with its own settings. Put this near the top of the script:
```bash
scancel --me --name="$SLURM_JOB_NAME" --state=PENDING
```
The first copy to start cancels its still-pending siblings; `--state=PENDING` means it never touches itself or anything running. A job can call Slurm commands just like you can from a login node.
```bash
sbatch --time=04:00:00 --constraint=a40  embeddings/carc/pretrain_pilot.sbatch --tokens 350M ...
sbatch --time=04:00:00 --constraint=l40s embeddings/carc/pretrain_pilot.sbatch --tokens 450M ...
sbatch --time=04:00:00 --constraint=a100 embeddings/carc/pretrain_pilot.sbatch --tokens 600M ...
```
Caveats: it also cancels any *other* pending job of yours with that name, so give unrelated jobs different names. If two copies start within the same few seconds, both run; `scancel` the slower one. A single `--constraint="a40|l40s|a100"` job is simpler when the settings don't depend on the card.

---

## 11. Scaling to ~30B

Goal: move HyperSteer from Gemma-2-2B to a ~30B base model (Gemma-2-27B? Qwen3-30B? **TBD**).

**Memory, rough:** bf16 weights ≈ 2 bytes/param → 27B ≈ **54 GB**, 30B ≈ **60 GB**, before activations. The base model is frozen, but the LM loss backprops through every layer after the steering layer, so those activations must be kept. Plus hypernetwork params + Adam state.

| Setup | Total VRAM | Fits 27–30B bf16? |
|---|---|---|
| 1× A40 / L40S | 48 GB | No |
| 1× A100-40GB | 40 GB | No |
| 1× A100-80GB | 80 GB | Weights yes; training very tight (small batch, grad checkpointing) |
| 2× A100-80GB (1 node) | 160 GB | Likely yes, with model sharding |
| 3× L40S (1 node) | 144 GB | Likely yes, with model sharding |
| 2× A40 (1 node) | 96 GB | Tight |

**Open questions / TODO:**
- [ ] Which 30B model exactly? (Sets layer count, hidden size, gating/licensing on HF.)
- [ ] How many A100s are 80 GB (`sinfo` in [§4](#4-running-commands-non-interactively)).
- [ ] Does the HyperSteer code support multi-GPU / `device_map="auto"` sharding? *(not checked)*
- [ ] Queue wait for 2×A100-80 or 3×L40S nodes in practice.
- [ ] SU budget: a multi-GPU 30B run burns ~5k SU/h; agree on a budget before launching on a shared class account.
- [ ] Validate everything at 2B on CARC first (smoke on `debug`), then scale.

---

## 12. Status / next steps

- [x] VPN working
- [x] SSH key login (`ssh carc`)
- [x] Account + SU budget confirmed (`xiangren_1987`)
- [x] GPU inventory
- [ ] Write `scripts/carc_train.sh`, `carc_setup.sh`, `carc_smoke.sbatch`, `carc_large.sbatch`, `push_to_carc.sh` (new files; RunPod scripts untouched)
- [ ] Push code, `uv sync`, pre-download Gemma-2-2B + concept16k to `/scratch1`
- [ ] Smoke test on `debug` (A40, ≤1 h), check SU charge with `jobinfo`
- [ ] Plan the 30B run ([§11](#11-scaling-to-30b))

---

## Sources

- [Discovery getting started](https://www.carc.usc.edu/user-guides/hpc-systems/discovery/getting-started-discovery): login, VPN, transfer nodes
- [VPN setup](https://www.carc.usc.edu/user-guides/quick-start-guides/anyconnect-vpn-setup)
- [Running jobs](https://www.carc.usc.edu/user-guides/hpc-systems/using-our-hpc-systems/running-jobs): sbatch/salloc, time limits
- [GPU programming](https://www.carc.usc.edu/user-guides/advanced-hpc-programming/gpu-programming): GPU types, constraints, SU rates, 36-GPU cap
- [Discovery resource overview](https://www.carc.usc.edu/user-guides/hpc-systems/discovery/resource-overview-discovery)
- [Storage file systems](https://www.carc.usc.edu/user-guides/research-data-management/storage-file-systems): quotas, backups, purge policy
- [Allocation types](https://www.carc.usc.edu/user-guides/project-and-allocation-management/allocation-types-and-requests)
- [SSH keys forum guide](https://hpc-discourse.usc.edu/t/how-to-set-up-ssh-keys-for-faster-login-login-w-o-password/128)
- [DNS-on-VPN fix](https://hpc-discourse.usc.edu/t/ssh-could-not-resolve-hostname-discovery-usc-edu/730)
- [Slurm templates](https://www.carc.usc.edu/user-guides/hpc-systems/using-our-hpc-systems/slurm-templates)
- [Slurm TRES / billing weights](https://slurm.schedmd.com/tres.html): sum of weighted TRES, integer truncation, `Mem=0.25G` semantics
- [Slurm resource limits](https://slurm.schedmd.com/resource_limits.html): `GrpTRESMins` usage decays with `PriorityDecayHalfLife`
- [slurm.conf](https://slurm.schedmd.com/slurm.conf.html): `AccountingStorageEnforce=safe`

Related: [[definitions]]
