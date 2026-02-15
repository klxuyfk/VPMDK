#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p reference
find reference -mindepth 1 -maxdepth 1 -exec rm -rf {} +

# If VPMDK is installed with pip, this is enough.
vpmdk > reference/run.log 2>&1

for file in CONTCAR OUTCAR OSZICAR vasprun.xml; do
  if [[ -f "${file}" ]]; then
    cp "${file}" "reference/${file}"
  fi
done

rm -f CONTCAR OUTCAR OSZICAR vasprun.xml

echo "relax_chgnet finished: reference/ updated"
