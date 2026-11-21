# Hackathon resources

General START Hack Munich 2026 infrastructure, not specific to any one challenge. Full step-by-step guide with
screenshots: **AMD GPU Handbook** in the guidebook —
https://munichstart.notion.site/Guidebook-START-HackMunich-2026-4167680686328172b9dadd06d9cf0fb1

## Does our challenge (Feyer · Faster Physics) need any of this?

**No.** The Differometor benchmark is explicitly CPU-only (`--cpus 1`, no GPU required), and scoring compares
*your own* before/after runs on *matching hardware*. Burning AMD credits on a GPU box for this would be wasted
spend and, worse, could invalidate your before/after comparison if you benchmark on two different machines.

Use [`container.bash` / `container.ps1`](../container.bash) locally for the actual timed runs. The AMD cloud is
useful if:
- WSL2/Docker on a teammate's Windows laptop is being painful and you want one standard Linux dev box instead
  (16 real CPU cores, 55 GB RAM — more than enough for this workload). If you do this, **pick one instance and have
  everyone benchmark there only** — don't mix a cloud run with a local Docker run for the final comparison.
- You want to poke at something unrelated to the scored benchmark (e.g. an LLM-assisted profiling/refactoring
  helper, or a side experiment) during the hack.

## AMD Radeon Cloud — free GPUs

AMD is giving the hackathon **150 Radeon GPUs** for the whole event. Each GPU: **48 GB VRAM**, 16 cores, 55 GB RAM,
ROCm + PyTorch + vLLM preinstalled.

**100% free, 48 credits per person** (AMD's free GPU hours, no real money, no payment ever). 1 credit = 1 GPU for
1 hour — covers the full 24 hours.

### Getting access (3 steps)

1. **Sign in:** https://radeon-global.anruicloud.com/ → **Login with Email** → enter the code you get.
   ⚠️ Do **not** use "AMD Developer Program" login — that account won't get the hackathon credits.
2. **Fill in the form:** https://tally.so/r/2jKBye — use the **exact same email** you signed in with.
3. **Wait for the next full hour** — everyone from the form is added on the hour. Then click your name (top
   right) → you should see **48 credits** and the **Hackathon** pool.

### SSH key FIRST, then launch (if working from your laptop / VS Code)

A key added *after* launching an instance is not installed on it — you'd have to destroy and relaunch.

1. On your laptop: `ssh-keygen -t ed25519 -C "laptop"` (press Enter for defaults).
2. Print the public key:
   - macOS/Linux: `cat ~/.ssh/id_ed25519.pub`
   - Windows: `type $env:USERPROFILE\.ssh\id_ed25519.pub`
3. Radeon Cloud → your name (top right) → **Settings** → **New SSH key** → paste the line starting with
   `ssh-ed25519`.

### Launching a GPU

On **Home**, switch on **Customize**, then pick:
- Deploy type: **Notebook** · Size: **1 GPU** · Image: **vLLM · Latest**
- Resource pool: **Hackathon** (our reserved GPUs) · Storage: **Persistent `/workspace`** if offered ·
  Auto-delete: **1 day**

Click **Launch** → wait for "Your instance is ready" → **Open Notebook** (JupyterLab with terminal).

For SSH: wait until the card shows `ssh root@<host> -p <port>`, then connect from your terminal or VS Code
(Remote - SSH).

### Important

- A running GPU burns credits even when you're not actively using it — there's no pause. **Destroy the instance
  when you're done.**
- Only `/workspace` survives a delete. Push to git often.
- Don't `pip install torch` — it replaces the preinstalled AMD (ROCm) build. (Not an issue for us anyway: this
  challenge uses JAX, not PyTorch — `pip install -r requirements.txt` is independent of the preinstalled torch.)
- One instance per account. Stuck on "Waiting for a free GPU"? Check you picked the **Hackathon** resource pool,
  then retry.

### No GPU needed?

The **Token Factory** tab gives hosted models through an OpenAI-compatible API (key starts with `rc-`), without
spending your GPU credits. Useful for any LLM-assisted tooling you build (e.g. an AI-assisted profiler/explainer)
without touching your 48 GPU-hours.

Questions on either: ask in the hackathon's Discord/Slack channel.
