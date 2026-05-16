#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
REPO_ROOT=$(cd ../.. && pwd)

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-vpmdk}"

WORK_DIR=$(mktemp -d)
STAGE_DIR=$(mktemp -d)
cleanup() {
  rm -rf "${WORK_DIR}" "${STAGE_DIR:-}"
}
trap cleanup EXIT

for file in POSCAR INCAR BCAR; do
  cp "${file}" "${WORK_DIR}/${file}"
done

(
  cd "${WORK_DIR}"
  PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" python -m vpmdk > run.log 2>&1
)

for file in CONTCAR OUTCAR OSZICAR vasprun.xml CHGCAR run.log; do
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

echo "chgcar_charge3net finished: reference/ updated"
