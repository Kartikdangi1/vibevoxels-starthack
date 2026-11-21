"""Run a command with an RSS cap and timeout; print peak RSS. Usage: guard.py CAP_MB TIMEOUT_S cmd...

Polls the summed RSS of the child's process tree every 0.2 s (via ps) and kills the whole
group if it exceeds CAP_MB, or if system free memory (memory_pressure) drops below 10 %.
"""
import os, signal, subprocess, sys, time

cap_mb, timeout = float(sys.argv[1]), float(sys.argv[2])
cmd = sys.argv[3:]
p = subprocess.Popen(cmd, start_new_session=True)
peak, t0, reason = 0.0, time.time(), None


def tree_rss_mb(root):
    out = subprocess.run(["ps", "-Ao", "pid=,ppid=,rss="], capture_output=True, text=True).stdout
    kids, rss = {}, {}
    for line in out.split("\n"):
        f = line.split()
        if len(f) == 3:
            pid, ppid, r = map(int, f)
            kids.setdefault(ppid, []).append(pid); rss[pid] = r
    total, stack = 0, [root]
    while stack:
        q = stack.pop(); total += rss.get(q, 0); stack += kids.get(q, [])
    return total / 1024


def free_pct():
    try:
        out = subprocess.run(["memory_pressure"], capture_output=True, text=True, timeout=5).stdout
        return int(out.strip().split("\n")[-1].split(":")[-1].strip().rstrip("%"))
    except Exception:
        return 100


n = 0
while p.poll() is None:
    m = tree_rss_mb(p.pid); peak = max(peak, m); n += 1
    if m > cap_mb:
        reason = f"RSS {m:.0f} MB > cap {cap_mb:.0f} MB"
    elif time.time() - t0 > timeout:
        reason = f"timeout {timeout:.0f} s"
    elif n % 10 == 0 and free_pct() < 10:
        reason = "system free memory < 10 %"
    if reason:
        os.killpg(p.pid, signal.SIGKILL); p.wait(); break
    time.sleep(0.2)
print(f"[guard] exit={p.returncode} peak_rss={peak:.0f}MB wall={time.time()-t0:.1f}s" + (f" KILLED: {reason}" if reason else ""),
      file=sys.stderr, flush=True)
sys.exit(p.returncode if not reason else 99)
