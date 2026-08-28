# Stage 1 P0 environment materialization

The frozen target prefix is `.conda/stage1-p0`. The package versions in
`environment/stage1-p0.yml` mirror the prefix that passed `pip check` on the
training host. This is a Torch 2.6.0 / CUDA 12.4 runtime, not the separate
Torch 2.8 seed environment used during early inspection.

Use two installation phases for a clean rebuild. FlashAttention must not be
built from its source distribution during the first pip solve because its
extension ABI must match the already-installed Torch build.

1. Create the Conda prefix and install the phase-1 requirements (the YAML stays
   the canonical declaration of the final state):

   ```bash
   conda create --prefix .conda/stage1-p0 --channel defaults \
     python=3.11.15 pip=26.0.1 setuptools=82.0.1 wheel=0.46.3
   .conda/stage1-p0/bin/python -m pip install \
     --requirement environment/stage1-p0-phase1.txt
   ```

2. After `import torch` reports `2.6.0+cu124` and
   `torch.compiled_with_cxx11_abi()` reports `False`, install the exact phase-2
   wheel without dependencies:

   ```bash
   .conda/stage1-p0/bin/python -m pip install --no-deps \
     'https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl#sha256=58853b28a5a926cae14402bfd8d4d93a45ebf8f9e79533f37ab09d0d77a99c05'
   ```

Before capture, run:

```bash
.conda/stage1-p0/bin/python -m pip check
.conda/stage1-p0/bin/python -c 'import torch, deepspeed, flash_attn, transformers; print(torch.__version__, torch.version.cuda, flash_attn.__version__, transformers.__version__)'
```

The YAML is the declarative final-state specification. The immutable
environment target additionally records every Python distribution and every
Conda package build/checksum, so a different transitive solve produces a new
`environment_build_id`.
