from __future__ import annotations

import vpmdk


def test_xdatcar_includes_lattice_when_cell_changes(tmp_path, load_atoms):
    atoms = load_atoms()
    target = tmp_path / "XDATCAR"

    vpmdk._XDATCAR_STATE.clear()
    vpmdk._write_xdatcar_step(target, atoms, 0)

    atoms.set_cell(atoms.get_cell().array * 1.01, scale_atoms=False)
    vpmdk._write_xdatcar_step(target, atoms, 1)

    lines = target.read_text().splitlines()

    assert lines[0] == lines[10]  # comment is repeated for each POSCAR block
    assert lines[2].strip() != lines[12].strip()  # lattice updated
    assert sum(1 for line in lines if line.startswith("Direct configuration=")) == 2


def test_xdatcar_retains_compact_format_for_static_cell(tmp_path, load_atoms):
    atoms = load_atoms()
    target = tmp_path / "XDATCAR"

    vpmdk._XDATCAR_STATE.clear()
    vpmdk._write_xdatcar_step(target, atoms, 0)
    vpmdk._write_xdatcar_step(target, atoms, 1)

    lines = target.read_text().splitlines()

    assert lines.count(lines[0]) == 1  # header appears once for static lattices
    assert len(lines) == 13  # 7 header + 2* (config + positions) for 2 atoms
    assert lines[10].startswith("Direct configuration=")
    assert sum(1 for line in lines if line.startswith("Direct configuration=")) == 2
