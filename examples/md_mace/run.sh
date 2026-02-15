#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p reference

MODEL_PATH=$(awk -F= '$1=="MODEL" {print $2}' BCAR | xargs)
if [[ -z "${MODEL_PATH}" || "${MODEL_PATH}" == "PATH_TO_MACE_MODEL" ]]; then
  echo "Set MODEL in BCAR before running (example: MODEL=./model/mace.model)" >&2
  exit 1
fi

find reference -mindepth 1 -maxdepth 1 -exec rm -rf {} +

# If VPMDK is installed with pip, this is enough.
vpmdk > reference/run.log 2>&1

for file in CONTCAR OUTCAR OSZICAR XDATCAR vasprun.xml; do
  if [[ -f "${file}" ]]; then
    cp "${file}" "reference/${file}"
  fi
done

rm -f CONTCAR OUTCAR OSZICAR XDATCAR vasprun.xml

echo "md_mace finished: reference/ updated"
