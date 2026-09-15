#!/usr/bin/env bash
# find_free_gpus_af3.sh -- stampa gli ID delle GPU fisiche con almeno
# MIN_FREE_GB gigabyte liberi in questo momento, uno per riga.
#
# Stessa soglia di default (21.0 GB) usata da get_available_gpus_with_memory()
# in tutti gli script Python del progetto (utils/gpu_utils.py) -- lo stesso
# criterio, solo utilizzabile anche a livello di shell PRIMA di lanciare
# python (necessario per poter impostare CUDA_VISIBLE_DEVICES prima
# dell'import di torch, il fix di isolamento GPU gia' validato in questo
# progetto per AF3).
#
# Uso:
#   bash find_free_gpus_af3.sh            # soglia default 21 GB
#   bash find_free_gpus_af3.sh 15         # soglia custom 15 GB
#   FREE_GPU=$(bash find_free_gpus_af3.sh | head -1)   # prendi solo la prima
set -euo pipefail
MIN_FREE_GB="${1:-21}"
MIN_FREE_MIB=$(( MIN_FREE_GB * 1024 ))

nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
  | awk -F',' -v min="${MIN_FREE_MIB}" '{gsub(/ /,"",$2); if ($2+0 >= min) print $1}'
