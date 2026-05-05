#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

cd "$(dirname "$0")"
REPO_ROOT=$(cd ../.. && pwd)
DEFAULT_PYTHON=/home/nei/miniconda3/envs/codex_nequip/bin/python
if [[ -z "${PYTHON:-}" && -x "${DEFAULT_PYTHON}" ]]; then
  PYTHON_BIN=${DEFAULT_PYTHON}
else
  PYTHON_BIN=${PYTHON:-python}
fi
PHONOPY_DIM=${PHONOPY_DIM:-"2 2 2"}
NEQUIP_MODEL=${NEQUIP_MODEL:-/mnt/d/lin_temp/codex/nequip/NequIP-OAM-L-0.1.nequip.pth}

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

NEQUIP_MODEL=$(resolve_path "${NEQUIP_MODEL}")
if [[ ! -f "${NEQUIP_MODEL}" ]]; then
  echo "NequIP model not found: ${NEQUIP_MODEL}" >&2
  echo "Set NEQUIP_MODEL=/path/to/model.pth or edit BCAR before running." >&2
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
sed -i "s|^MODEL=.*|MODEL=${NEQUIP_MODEL}|" "${WORK_DIR}/BCAR"

(
  cd "${WORK_DIR}"
  phonopy -d --dim "${PHONOPY_DIM}" > phonopy_displacements.log 2>&1

  poscars=(POSCAR-[0-9]*)
  if [[ ${#poscars[@]} -eq 0 ]]; then
    echo "phonopy did not create any POSCAR-### displacement files" >&2
    exit 1
  fi

  mkdir -p displacements
  for poscar in "${poscars[@]}"; do
    index=${poscar#POSCAR-}
    calc_dir="displacements/disp-${index}"
    mkdir -p "${calc_dir}"
    cp "${poscar}" "${calc_dir}/POSCAR"
    cp INCAR BCAR "${calc_dir}/"
    (
      cd "${calc_dir}"
      PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" -m vpmdk > vpmdk.log 2>&1
    )
  done

  phonopy -f displacements/disp-*/vasprun.xml > phonopy_forces.log 2>&1
)

rm -rf output
mkdir -p output/displacements

for file in SPOSCAR phonopy_disp.yaml FORCE_SETS phonopy_displacements.log phonopy_forces.log; do
  if [[ -f "${WORK_DIR}/${file}" ]]; then
    cp "${WORK_DIR}/${file}" "output/${file}"
  fi
done

for calc_dir in "${WORK_DIR}"/displacements/disp-*; do
  if [[ -d "${calc_dir}" ]]; then
    name=$(basename "${calc_dir}")
    mkdir -p "output/displacements/${name}"
    for file in POSCAR CONTCAR OUTCAR OSZICAR vasprun.xml vpmdk.log; do
      if [[ -f "${calc_dir}/${file}" ]]; then
        cp "${calc_dir}/${file}" "output/displacements/${name}/${file}"
      fi
    done
  fi
done

echo "phonopy_displacement_force_sets finished: output/ updated"
