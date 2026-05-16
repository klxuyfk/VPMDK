#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
REPO_ROOT=$(cd ../.. && pwd)
DEFAULT_PYTHON=/home/nei/miniconda3/envs/codex_sevenn/bin/python
if [[ -z "${PYTHON:-}" && -x "${DEFAULT_PYTHON}" ]]; then
  PYTHON_BIN=${DEFAULT_PYTHON}
else
  PYTHON_BIN=${PYTHON:-python}
fi
SEVENNET_MODEL=${SEVENNET_MODEL:-7net-0}

if ! command -v phonopy >/dev/null 2>&1; then
  echo "phonopy is not on PATH. Install phonopy in the active environment first." >&2
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
sed -i "s|^MODEL=.*|MODEL=${SEVENNET_MODEL}|" "${WORK_DIR}/BCAR"

(
  cd "${WORK_DIR}"
  PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" -m vpmdk > vpmdk.log 2>&1
  phonopy --fc vasprun.xml > phonopy_fc.log 2>&1
)

rm -rf output
mkdir -p output

for file in CONTCAR OUTCAR OSZICAR vasprun.xml FORCE_CONSTANTS vpmdk.log phonopy_fc.log; do
  if [[ -f "${WORK_DIR}/${file}" ]]; then
    cp "${WORK_DIR}/${file}" "output/${file}"
  fi
done

echo "phonopy_vasp_finite_difference finished: output/ updated"
