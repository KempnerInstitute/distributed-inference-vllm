# Environments

This directory contains reproducible runtime environment definitions for executing vLLM inference workflows.

## Available Environments

### uv Environments

| Name | Python | vLLM | CUDA | Notes |
|------|--------|------|------|-------|
| [u260304_vllm](uv/u260304_vllm/) | 3.12.11 | 0.11.2 | 12.9.1 |  |

### Conda Environments

*No conda environments available yet.*

### Docker Environments

*No docker environments available yet.*

### Singularity Environments

*No singularity environments available yet.*

---

## Purpose

Environments provide the foundation for all workflows, reports, and workshops in this repository. Each environment is a complete, reproducible setup that includes:

- Exact package versions with lock files
- Installation instructions
- Verification procedures
- Hardware and software specifications
- Troubleshooting guidance

## Directory Structure

```
envs/
  README.md (this file)
  conda/          # Conda environments
  docker/         # Docker containers
  uv/             # uv-based Python environments
  singularity/    # Singularity containers
```

## Naming Convention

Environment folders follow this naming pattern:

```
<type-prefix><date>_<short-description>
```

**Environment Type Prefixes:**

| Prefix | Type        | Example               |
|--------|-------------|-----------------------|
| `c`    | Conda       | `c260304_vllm`        |
| `u`    | uv          | `u260304_vllm`        |
| `d`    | Docker      | `d260304_cuda124`     |
| `s`    | Singularity | `s260304_vllm`        |

**Date Format:** `YYMMDD` (year, month, day)

## Using Environments

Each environment directory contains:

1. **README.md** - Complete setup guide with:
   - Specifications
   - Installation instructions
   - Verification steps
   - Usage examples
   - Troubleshooting

2. **Environment Definition Files:**
   - Conda: `environment.yml`
   - uv: `pyproject.toml`, `uv.lock`
   - Docker: `Dockerfile`
   - Singularity: `*.def`

3. **Additional Resources:** (when applicable)
   - Test scripts
   - Setup scripts
   - Configuration files
   - Documentation

## Environment Versioning

### For Minor Updates (Bug Fixes/Patches)
Keep the same date and append version suffix:
- Example: `u260304_vllm` → `u260304_vllm_v1`

### For Major Changes
Create a new environment with a new date:
- New dependencies added/removed
- Version changes (Python, CUDA, PyTorch, vLLM)
- Major configuration changes
- Example: `u260304_vllm` → `u260315_vllm`

Document the relationship in the new environment's README under a "History" section.

## Contributing

To contribute a new environment, see the detailed guidelines in [CONTRIBUTING.md](../CONTRIBUTING.md).

**Quick checklist:**
1. Create appropriately named directory
2. Include all environment definition files
3. Write comprehensive README
4. Test environment creation from scratch
5. Submit pull request

## External Resources

**Important:** Environment files (conda .tar.gz, Docker images, Singularity .sif files) should NOT be committed to this repository. Instead:

- Document exact build/creation steps
- Provide external links (Google Drive, AWS S3, Docker Hub)
- Include checksums for verification
- Docker images should reference Docker Hub repositories

## Questions?

For questions about environments or contributing:
- Check existing environments for examples
- Review [CONTRIBUTING.md](../CONTRIBUTING.md)
- Open an issue for discussion
- Contact repository maintainers
