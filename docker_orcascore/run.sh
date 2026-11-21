#!/usr/bin/env bash
# Runs a command of the repository inside the Docker image, in the background by default.
#
# Usage:
#   ./run.sh [-i] <container_name> <gpu> <folder> <command...>
#     -i            interactive (attached to the terminal) instead of background
#     container_name  name of the container (to follow the log / stop it)
#     gpu           GPU index on the host (0 or 1 on the curupira); inside the container it is cuda:0
#     folder        TransMorph_affine or TransMorph (working directory of the command)
#
# Example:
#   ./run.sh afim_v1 0 TransMorph_affine python train_TransMorph_affine.py \
#       --dataset_dir /workspace/data/dataset_original_comum --name afim_v1
#
# Paths: edit the defaults below or export them before calling (e.g. DATA_DIR=/outro ./run.sh ...).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(dirname "$SCRIPT_DIR")}"         # repository (mounted at /workspace/repo)
DATA_DIR="${DATA_DIR:-/home/tcc-repository/orcascore/datasets}"  # folder with the datasets (mounted at /workspace/data)
IMAGE="${IMAGE:-transmorph_orcascore}"

interactive=0
if [[ "${1:-}" == "-i" ]]; then
    interactive=1
    shift
fi
if [[ $# -lt 4 ]]; then
    sed -n '2,17p' "$0"
    exit 1
fi
name="$1"; gpu="$2"; folder="$3"; shift 3

if docker ps -a --format '{{.Names}}' | grep -qx "$name"; then
    echo "A container named '$name' already exists. Remove it with: docker rm $name" >&2
    exit 1
fi

docker_args=(
    --name "$name"
    --gpus "device=$gpu"
    --ipc=host
    --user "$(id -u):$(id -g)"
    -e HOME=/tmp
    -v "$REPO_DIR":/workspace/repo
    -v "$DATA_DIR":/workspace/data
    -w "/workspace/repo/$folder"
)

if [[ $interactive -eq 1 ]]; then
    docker run --rm -it "${docker_args[@]}" "$IMAGE" "$@"
else
    docker run -d "${docker_args[@]}" "$IMAGE" "$@"
    echo "Started '$name' in the background (GPU $gpu)."
    echo "  follow the log:  docker logs -f $name"
    echo "  stop:            docker stop $name"
    echo "  remove (after it ends): docker rm $name"
fi
