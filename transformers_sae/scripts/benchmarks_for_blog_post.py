import os
import sys

# Import the methods list
from scripts.methods_for_blog_post import TUNED_ENCODER_METHODS

if not TUNED_ENCODER_METHODS:
    print("No TUNED_ENCODER_METHODS defined.", file=sys.stderr)
    sys.exit(1)

run_script = "./run.sh"
cmd = [run_script, "benchmark_gemma"]
for method in TUNED_ENCODER_METHODS + ["baseline"]:
    cmd.extend(["-m", method])

# Execute in the calling shell (i.e., replace current process)
os.execvp(run_script, cmd)