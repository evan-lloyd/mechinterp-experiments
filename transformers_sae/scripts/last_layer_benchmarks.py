import os
import sys

# This is code to set up for an analysis that didn't make it into the blog post, which was going to
# show that the SAE representation was theoretically adequate for question-answering since
# models get above-chance performance when only replacing the final layer. However I instead opted
# to mix in some CQA examples in the KL-tuning phase, which was sufficient to show variability
# across training methods.

# Import the methods list
from scripts.methods_for_blog_post import TUNED_ENCODER_METHODS

if not TUNED_ENCODER_METHODS:
    print("No TUNED_ENCODER_METHODS defined.", file=sys.stderr)
    sys.exit(1)

run_script = "./run.sh"
cmd = [run_script, "benchmark_gemma"]
for method in TUNED_ENCODER_METHODS + ["baseline"]:
    cmd.extend(["-m", method])

cmd.extend(["-l", "25"])

# Execute in the calling shell (i.e., replace current process)
os.execvp(run_script, cmd)
