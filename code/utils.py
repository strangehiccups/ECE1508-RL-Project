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
    """Launch a script in a separate process (used for scripts that open the MuJoCo viewer).
    On macOS the viewer must run under `mjpython`, which ships with mujoco next to the interpreter."""
    assert Path(script).exists()
    exe = sys.executable
    if sys.platform == "darwin":
        mjpython = Path(sys.executable).with_name("mjpython")
        assert mjpython.exists(), "mjpython not found next to the Python interpreter; is mujoco installed?"
        exe = str(mjpython)
    subprocess.Popen([exe, script])