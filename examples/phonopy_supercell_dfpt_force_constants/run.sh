#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
REPO_ROOT=$(cd ../.. && pwd)
DEFAULT_PYTHON=/home/nei/miniconda3/envs/codex_orb/bin/python
if [[ -z "${PYTHON:-}" && -x "${DEFAULT_PYTHON}" ]]; then
  PYTHON_BIN=${DEFAULT_PYTHON}
else
  PYTHON_BIN=${PYTHON:-python}
fi
PHONOPY_DIM=${PHONOPY_DIM:-"2 2 2"}
ORB_MODEL_PATH=${ORB_MODEL_PATH:-/mnt/d/lin_temp/codex/orb/orb-v3-conservative-20-omat-20250404.ckpt}

resolve_path() {
  local path=$1
  if [[ "${path}" == ~* ]]; then
    path="${path/#\~/${HOME}}"
  fi
  if [[ "${path}" != /* ]]; then
    path="${PWD}/${path}"
  fi
  printf '%s\n' "${path}"
}

if ! command -v phonopy >/dev/null 2>&1; then
  echo "phonopy is not on PATH. Install phonopy in the active environment first." >&2
  exit 1
fi

ORB_MODEL_PATH=$(resolve_path "${ORB_MODEL_PATH}")
if [[ ! -f "${ORB_MODEL_PATH}" ]]; then
  echo "ORB checkpoint not found: ${ORB_MODEL_PATH}" >&2
  echo "Set ORB_MODEL_PATH=/path/to/orb.ckpt or edit BCAR before running." >&2
  exit 1
fi

WORK_DIR=$(mktemp -d)
cleanup() {
  rm -rf "${WORK_DIR}"
}
trap cleanup EXIT

for file in POSCAR INCAR BCAR; do
  cp "${file}" "${WORK_DIR}/${file}"
done
sed -i "s|^MODEL=.*|MODEL=${ORB_MODEL_PATH}|" "${WORK_DIR}/BCAR"

(
  cd "${WORK_DIR}"
  phonopy -d --dim "${PHONOPY_DIM}" > phonopy_supercell.log 2>&1
  cp SPOSCAR POSCAR
  PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" -m vpmdk > vpmdk.log 2>&1
  phonopy --fc vasprun.xml > phonopy_fc.log 2>&1
)

rm -rf output
mkdir -p output

for file in SPOSCAR phonopy_disp.yaml CONTCAR OUTCAR OSZICAR vasprun.xml FORCE_CONSTANTS vpmdk.log phonopy_supercell.log phonopy_fc.log; do
  if [[ -f "${WORK_DIR}/${file}" ]]; then
    cp "${WORK_DIR}/${file}" "output/${file}"
  fi
done

echo "phonopy_supercell_dfpt_force_constants finished: output/ updated"
