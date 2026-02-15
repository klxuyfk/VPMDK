#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

MODEL_PATH=$(
  awk '
    {
      line = $0
      sub(/[#!].*$/, "", line)
      if (index(line, "=") == 0) {
        next
      }
      key = substr(line, 1, index(line, "=") - 1)
      val = substr(line, index(line, "=") + 1)
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", key)
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", val)
      if (toupper(key) == "MODEL") {
        print val
        exit
      }
    }
  ' BCAR
)
if [[ -z "${MODEL_PATH}" || "${MODEL_PATH}" == "PATH_TO_MACE_MODEL" ]]; then
  echo "Set MODEL in BCAR before running (example: MODEL=./model/mace.model)" >&2
  exit 1
fi

if [[ "${MODEL_PATH}" == ~* ]]; then
  MODEL_PATH="${MODEL_PATH/#\~/${HOME}}"
fi
if [[ "${MODEL_PATH}" != /* ]]; then
  MODEL_PATH="${PWD}/${MODEL_PATH}"
fi

WORK_DIR=$(mktemp -d)
STAGE_DIR=$(mktemp -d)
cleanup() {
  rm -rf "${WORK_DIR}" "${STAGE_DIR:-}"
}
trap cleanup EXIT

for file in POSCAR INCAR BCAR; do
  if [[ -f "${file}" ]]; then
    cp "${file}" "${WORK_DIR}/${file}"
  fi
done

echo "MODEL=${MODEL_PATH}" >> "${WORK_DIR}/BCAR"

# If VPMDK is installed with pip, this is enough.
(
  cd "${WORK_DIR}"
  vpmdk > run.log 2>&1
)

for file in CONTCAR OUTCAR OSZICAR XDATCAR vasprun.xml run.log; do
  if [[ -f "${WORK_DIR}/${file}" ]]; then
    cp "${WORK_DIR}/${file}" "${STAGE_DIR}/${file}"
  fi
done

rm -rf reference.new
mv "${STAGE_DIR}" reference.new
STAGE_DIR=""
if [[ -d reference ]]; then
  rm -rf reference.prev
  mv reference reference.prev
fi
mv reference.new reference
rm -rf reference.prev

echo "md_mace finished: reference/ updated"
