import subprocess
import argparse
# Argument Parser
parser = argparse.ArgumentParser(description="Check GPU usage and report idle GPUs.")
parser.add_argument("--t", type=float, default=20.0, help="Threshold percentage below which GPU is considered idle")
args = parser.parse_args()

# Run nvidia-smi command
out = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout

# Process output
out2 = out.split("\n")[8:]
splitter = out2[3]
out3 = "\n".join(out2).split(splitter)[:-1]

# Check GPU usage
for x in out3:
    gpu = ((x.split("|"))[1].replace("Off", ""))
    t = ((x.split("|"))[6].replace("Off", "").replace("MiB", "").split("/"))
    ans = (int(t[0]) / int(t[1])) * 100
    if ans < args.t:
        print(f"{gpu[:-5]}--->   GPU is idle ({ans:.2f}%)")
