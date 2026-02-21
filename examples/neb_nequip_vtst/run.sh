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
if [[ -z "${MODEL_PATH}" || "${MODEL_PATH}" == "PATH_TO_NEQUIP_MODEL" ]]; then
  echo "Set MODEL in BCAR before running (example: MODEL=./model/nequip_model.pth)" >&2
  exit 1
fi

if [[ "${MODEL_PATH}" == ~* ]]; then
  MODEL_PATH="${MODEL_PATH/#\~/${HOME}}"
fi
if [[ "${MODEL_PATH}" != /* ]]; then
  MODEL_PATH="${PWD}/${MODEL_PATH}"
fi

if [[ -n "${NEQUIP_SOURCE:-}" && "${NEQUIP_SOURCE}" != /* ]]; then
  NEQUIP_SOURCE="${PWD}/${NEQUIP_SOURCE}"
fi

WORK_DIR=$(mktemp -d)
STAGE_DIR=$(mktemp -d)
VTST_TMP=$(mktemp -d)
cleanup() {
  rm -rf "${WORK_DIR}" "${STAGE_DIR:-}" "${VTST_TMP}"
}
trap cleanup EXIT

for file in POSCAR INCAR BCAR; do
  if [[ -f "${file}" ]]; then
    cp "${file}" "${WORK_DIR}/${file}"
  fi
done
for image in 00 01 02; do
  if [[ -d "${image}" ]]; then
    cp -a "${image}" "${WORK_DIR}/${image}"
  fi
done

echo "MODEL=${MODEL_PATH}" >> "${WORK_DIR}/BCAR"

(
  cd "${WORK_DIR}"
  if [[ -n "${NEQUIP_SOURCE:-}" ]]; then
    PYTHONPATH="${NEQUIP_SOURCE}:${PYTHONPATH:-}" vpmdk > run.log 2>&1
  else
    vpmdk > run.log 2>&1
  fi
)

(
  cd "${VTST_TMP}"
  curl -fsSLO https://theory.cm.utexas.edu/code/vtstscripts.tgz
  tar -xzf vtstscripts.tgz
)
VTST_DIR=$(find "${VTST_TMP}" -maxdepth 1 -type d -name 'vtstscripts-*' | head -n1)
(
  cd "${WORK_DIR}"
  PERL5LIB="${VTST_DIR}" perl "${VTST_DIR}/nebresults.pl" > vtst_nebresults.log 2>&1
)

for file in OUTCAR OSZICAR vasprun.xml neb.dat nebef.dat spline.dat exts.dat mep.eps movie movie.vasp run.log vtst_nebresults.log; do
  if [[ -f "${WORK_DIR}/${file}" ]]; then
    cp "${WORK_DIR}/${file}" "${STAGE_DIR}/${file}"
  fi
done

if [[ -f "${WORK_DIR}/vaspgr/vaspout1.eps" ]]; then
  mkdir -p "${STAGE_DIR}/vaspgr"
  cp "${WORK_DIR}/vaspgr/vaspout1.eps" "${STAGE_DIR}/vaspgr/vaspout1.eps"
fi

for image in 00 01 02; do
  mkdir -p "${STAGE_DIR}/${image}"
  for file in CONTCAR OSZICAR OUTCAR.gz vasprun.xml POSCAR.vasp POSCAR.xyz fe.dat; do
    if [[ -f "${WORK_DIR}/${image}/${file}" ]]; then
      cp "${WORK_DIR}/${image}/${file}" "${STAGE_DIR}/${image}/${file}"
    fi
  done
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

echo "neb_nequip_vtst finished: reference/ updated"
