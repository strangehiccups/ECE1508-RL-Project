"""
Train all three v2 variants with one command (runs them in parallel).

  python train_all.py --smoke          # 1-2 minute speed test; prints time estimates, cleans up after itself
  python train_all.py                  # the three real runs: res_z, res_noz, e2e_z
  python train_all.py --only res_z     # just one (or two) of them

This window prints a short status every minute. Each run's full output is saved in logs/<name>.out.
Ctrl+C stops everything. On a Mac the computer is kept awake while this runs (lid must stay open).

If only Phase 2 of a run was interrupted, finish it with:   python train.py --phase 2 --name res_z
"""
import argparse
import glob
import os
import platform
import re
import shutil
import subprocess
import sys
import time

RUNS = {
    "res_z":   ["--phase", "all", "--mode", "residual"],                  # residual RMA (main method)
    "res_noz": ["--phase", "1", "--mode", "residual", "--no-z"],          # residual without z
    "e2e_z":   ["--phase", "all", "--mode", "e2e", "--iters", "800"],     # end-to-end with z
}
SMOKE = {"smoke": ["--phase", "all", "--mode", "residual", "--iters", "3", "--iters2", "2"]}
STEPS_PER_ITER = 2048 * 24


def last_progress(path):
    try:
        with open(path, errors="ignore") as f:
            lines = [line.strip() for line in f if line.startswith("[P")]
        return lines[-1] if lines else "starting (first run compiles kernels, can take a few minutes)"
    except FileNotFoundError:
        return "starting"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="quick speed test")
    ap.add_argument("--only", nargs="+", choices=list(RUNS), help="run only these variants")
    args = ap.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    runs = SMOKE if args.smoke else {k: v for k, v in RUNS.items() if not args.only or k in args.only}

    if platform.system() == "Darwin" and shutil.which("caffeinate"):
        subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])      # keep the Mac awake until we exit

    env = dict(os.environ, OMP_NUM_THREADS="1", PYTHONUNBUFFERED="1")        # ~1 core per run, live logs
    procs = {}
    for name, extra in runs.items():
        log = open(f"logs/{name}.out", "w")
        cmd = [sys.executable, "-u", "train.py", "--name", name] + extra
        procs[name] = (subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env), log)
        print(f"started {name:8s} -> logs/{name}.out")

    t0 = time.time()
    try:
        while True:
            time.sleep(10 if args.smoke else 60)
            print(f"\n--- {(time.time() - t0) / 60:.0f} min ---")
            running = 0
            for name, (p, _) in procs.items():
                code = p.poll()
                running += code is None
                state = "running" if code is None else ("DONE" if code == 0 else f"FAILED (see logs/{name}.out)")
                print(f"{name:8s} {state:8s} {last_progress(f'logs/{name}.out')}")
            if running == 0:
                break
    except KeyboardInterrupt:
        print("\nstopping all runs ...")
        for p, _ in procs.values():
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.terminate()
    for _, log in procs.values():
        log.close()

    if args.smoke:
        text = open("logs/smoke.out", errors="ignore").read()
        sps = [int(s.replace(",", "")) for s in re.findall(r"\[P1 smoke\].*?([\d,]+) sps", text)]
        if sps:
            s = max(sps)
            print(f"\nspeed: about {s:,} steps/s for one run on its own (three in parallel will each be a bit slower)")
            print(f"estimate: 400 iterations ~ {400 * STEPS_PER_ITER / s / 60:.0f} min, "
                  f"800 iterations ~ {800 * STEPS_PER_ITER / s / 60:.0f} min, plus ~30-40 min for each Phase 2")
        else:
            print("\nthe speed test did not finish, see logs/smoke.out")
        for f in glob.glob("checkpoints/smoke*") + glob.glob("logs/smoke*"):
            os.remove(f)
    else:
        print("\nall runs finished. Models are in checkpoints/, training logs in logs/.")


if __name__ == "__main__":
    main()
