#!/usr/bin/env python3
"""
compute_running_stats.py

Parse a docking log and print running mean & stdev for DockQ and iRMSD.

Usage:
  python compute_running_stats.py path/to/log.txt
  # or: cat log.txt | python compute_running_stats.py
"""
import sys, re, math

# Regex to capture a single sample's metrics (the first 'dockq:' on the line)
SAMPLE_RE = re.compile(
    r"lrmsd:\s*([-\d.]+)\s*,\s*"
    r"irmsd:\s*([-\d.]+)\s*,\s*"
    r"crmsd:\s*([-\d.]+)\s*,\s*"
    r"dockq:\s*([-\d.]+)\s*;",
    re.IGNORECASE,
)

class RunningStats:
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0  # Sum of squared diffs from the mean

    def update(self, x: float):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    @property
    def stdev(self) -> float:
        if self.n < 2:
            return 0.0
        return math.sqrt(self.M2 / (self.n - 1))

def stream_lines():
    if not sys.stdin.isatty():
        for line in sys.stdin:
            yield line.rstrip("\n")
    elif len(sys.argv) > 1:
        with open(sys.argv[1], "r", encoding="utf-8") as f:
            for line in f:
                yield line.rstrip("\n")
    else:
        sys.exit("Provide a log file or pipe the log on stdin.")

def main():
    dockq_stats = RunningStats()
    irmsd_stats = RunningStats()
    k = 0

    print("#idx\trun_mean_DockQ\trun_std_DockQ\trun_mean_iRMSD\trun_std_iRMSD")

    for line in stream_lines():
        m = SAMPLE_RE.search(line)
        if not m:
            continue

        # groups: 1=lrmsd, 2=irmsd, 3=crmsd, 4=dockq
        irmsd = float(m.group(2))
        dockq = float(m.group(4))

        dockq_stats.update(dockq)
        irmsd_stats.update(irmsd)

        k += 1
        print(
            f"{k}\t"
            f"{dockq_stats.mean:.6f}\t{dockq_stats.stdev:.6f}\t"
            f"{irmsd_stats.mean:.6f}\t{irmsd_stats.stdev:.6f}"
        )

    if k == 0:
        sys.stderr.write("No samples found. Check that your log lines contain "
                         "'lrmsd: ..., irmsd: ..., crmsd: ..., dockq: ...;'\n")

if __name__ == "__main__":
    main()