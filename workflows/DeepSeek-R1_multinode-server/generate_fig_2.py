"""
Generates Figure 2 from the 'README_DeepSeekR1.md' file.

This script plots throughput scaling H100 GPU with default vLLM batching behavior.
"""

import matplotlib.pyplot as plt

# H100 data
h100_num_prompts = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536, 2048]
h100_throughput = [10.3, 19.6, 38.8, 78.4, 155.1, 307.1, 607.7, 1164.3, 2251.8, 4008.5, 5834.4, 5836.0, 5836.0]

# Create plot
plt.figure(figsize=(10, 6))
plt.plot(h100_num_prompts, h100_throughput, marker='o', label='16 × H100 80GB GPUs', linewidth=2, color='blue')

# Labels and title
plt.title('Throughput Scaling on H100 GPU (FP8)', fontsize=18)
plt.xlabel('Concurrent Prompts', fontsize=13)
plt.ylabel('Throughput (tokens/sec)', fontsize=13)
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.xscale('log', base=2)
plt.yscale('log', base=2)
plt.xticks(h100_num_prompts)
plt.legend()
plt.tight_layout()

# Show plot
plt.savefig("dsr1_throughput_scaling_default.png", dpi=300)
plt.show()
