# DeepSeek-R1-0528

DeepSeek-R1-0528 ([Hugging Face Page](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528)) is a minor upgrade to DeepSeek R1, enhancing reasoning and inference through algorithmic optimizations and increased compute. This version shows improvements in math, programming, and logic, rivaling top models like O3 and Gemini 2.5 Pro. 

For deployment instructions, follow the same steps as in the [DeepSeek-R1](README_DeepSeekR1.md) section, but change the model path as follows:

- In your curl command:

  ```bash
  "model": "/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models/DeepSeek-R1-0528"
  ```
- In your Python scripts:

  ```python
  MODEL_PATH = "/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models/DeepSeek-R1-0528"
  ```

- In your SLURM submission scripts:

  ```bash
  MODEL_PATH="/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models/DeepSeek-R1-0528"
  ```
