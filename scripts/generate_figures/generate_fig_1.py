"""
Generates Figure 1 from the 'README_DeepSeekR1.md' file.

This script plots throughput scaling on H100 and H200 GPUs.
"""

import matplotlib.pyplot as plt

# H100 data
h100_batched_tokens = [8192, 16384, 24576, 32768, 49152, 57344]
h100_throughput = [76, 300, 446, 607, 902, 1030]

# H200 data
h200_batched_tokens = [65536, 81920, 98304]
h200_throughput = [690, 921, 1022]

# Create plot
plt.figure(figsize=(10, 6))
plt.plot(h100_batched_tokens, h100_throughput, marker='o', label='16 x H100 80GB GPUs', linewidth=2, color='blue')
plt.plot(h200_batched_tokens, h200_throughput, marker='s', label='8 x H200 141GB GPUs', linewidth=2, color='red')

# Labels and title
plt.title('Throughput Scaling on H100 and H200 GPUs (FP8)', fontsize=18)
plt.xlabel('Max Batched Tokens', fontsize=13)
plt.ylabel('Throughput (tokens/sec)', fontsize=13)
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.xscale('log', base=2)
plt.yscale('log', base=2)
plt.xticks(h100_batched_tokens + h200_batched_tokens)
plt.legend()
plt.tight_layout()

# Show plot
plt.savefig("throughput_scaling_h100_h200.png", dpi=300)
plt.show()
