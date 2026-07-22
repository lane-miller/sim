# sim

## Environment setup

This repo uses a conda environment named `simenv`.

### Activate the environment

```bash
conda activate simenv
```

### If `simenv` doesn't exist yet

Check whether it's already installed:

```bash
conda env list
```

If it's missing, recreate it (adjust as needed depending on how it was originally created, e.g. from an `environment.yml`):

```bash
conda create -n simenv python=<version>
conda activate simenv
```

### Deactivate

```bash
conda deactivate
```
