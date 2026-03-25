# LAB 1.a — Connect to Compute Node & Set Up Environment

## Estimated Time
10–15 minutes

---

## Objective

By the end of this lab, you will:

- Connect to the Kempner cluster
- Allocate a GPU compute node
- Set up a Python environment for inference

---

## 1. Connect to the Cluster

From your local machine:

```bash
ssh <your_username>@login.rc.fas.harvard.edu
```

## 2. Allocate a GPU Compute Node

```bash
salloc -p kempner_h100 --gres=gpu:1 --mem=16G --time=01:00:00 (TBD)
```

## 3. Check GPU Availability

```bash
nvidia-smi
```

## 4. Set Up Python Environment

In order to set up Python environment, we have several options. You can see the list of available approaches in the [envs](../envs) directory. For this lab, we will use the TBD approach.

```bash
TBD
```
## 5. Verify Environment Setup

```bash
TBD
```

Done!
