#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p reference

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

if [[ -n "${NEQUIP_SOURCE:-}" ]]; then
  export PYTHONPATH="${NEQUIP_SOURCE}:${PYTHONPATH:-}"
fi

find reference -mindepth 1 -maxdepth 1 -exec rm -rf {} +

vpmdk > reference/run.log 2>&1

VTST_TMP=$(mktemp -d)
trap 'rm -rf "${VTST_TMP}"' EXIT
(
  cd "${VTST_TMP}"
  curl -fsSLO https://theory.cm.utexas.edu/code/vtstscripts.tgz
  tar -xzf vtstscripts.tgz
)
VTST_DIR=$(find "${VTST_TMP}" -maxdepth 1 -type d -name 'vtstscripts-*' | head -n1)
PERL5LIB="${VTST_DIR}" perl "${VTST_DIR}/nebresults.pl" > reference/vtst_nebresults.log 2>&1
for file in OUTCAR OSZICAR vasprun.xml neb.dat nebef.dat spline.dat exts.dat mep.eps movie movie.vasp; do
  if [[ -f "${file}" ]]; then
    cp "${file}" "reference/${file}"
  fi
done

mkdir -p reference/vaspgr
if [[ -f vaspgr/vaspout1.eps ]]; then
  cp vaspgr/vaspout1.eps reference/vaspgr/vaspout1.eps
fi

for image in 00 01 02; do
  mkdir -p "reference/${image}"
  for file in CONTCAR OSZICAR OUTCAR.gz vasprun.xml POSCAR.vasp POSCAR.xyz; do
    if [[ -f "${image}/${file}" ]]; then
      cp "${image}/${file}" "reference/${image}/${file}"
    fi
  done
done

if [[ -f 01/fe.dat ]]; then
  cp 01/fe.dat reference/01/fe.dat
fi

rm -f OUTCAR OSZICAR vasprun.xml neb.dat nebef.dat spline.dat exts.dat mep.eps movie movie.vasp
rm -rf vaspgr
for image in 00 01 02; do
  rm -f "${image}/CONTCAR" "${image}/OSZICAR" "${image}/OUTCAR" "${image}/OUTCAR.gz"
  rm -f "${image}/vasprun.xml" "${image}/POSCAR.vasp" "${image}/POSCAR.xyz" "${image}/fe.dat"
done

echo "neb_nequip_vtst finished: reference/ updated"
