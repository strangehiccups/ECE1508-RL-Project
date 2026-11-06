import sys, subprocess
from pathlib import Path

def run_streaming(script):
    assert Path(script).exists()
    p = subprocess.Popen([sys.executable, "-u", script],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1)
    try:
        for line in p.stdout:
            print(line, end="")
    except KeyboardInterrupt:
        p.terminate()
        print("\nTraining stopped. Last checkpoint is in policy.pt")
    finally:
        p.wait()

def run(script):
    assert Path(script).exists()
    subprocess.Popen([sys.executable, script])