#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
REPO_ROOT=$(cd ../.. && pwd)

if [[ -z "${VPMDK_CHARGE_SOURCE_DIR:-}" ]]; then
  echo "VPMDK_CHARGE_SOURCE_DIR is not set" >&2
  exit 1
fi

if [[ -z "${VPMDK_CHARGE_PYTHON:-}" ]]; then
  echo "VPMDK_CHARGE_PYTHON is not set" >&2
  exit 1
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-vpmdk}"

WORK_DIR=$(mktemp -d)
cleanup() {
  rm -rf "${WORK_DIR}"
}
trap cleanup EXIT

for file in POSCAR INCAR BCAR; do
  cp "${file}" "${WORK_DIR}/${file}"
done

(
  cd "${WORK_DIR}"
  PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" python -m vpmdk > run.log 2>&1
)

rm -rf output
mkdir -p output

for file in CONTCAR OUTCAR OSZICAR vasprun.xml CHGCAR run.log; do
  if [[ -f "${WORK_DIR}/${file}" ]]; then
    cp "${WORK_DIR}/${file}" "output/${file}"
  fi
done

echo "chgcar_charge3net finished: output/ updated"
