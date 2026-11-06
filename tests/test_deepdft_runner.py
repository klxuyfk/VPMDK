from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.neighborlist import neighbor_list

spec = importlib.util.spec_from_file_location(
    'deepdft_runner',
    Path(__file__).resolve().parents[1] / 'src/vpmdk_core/deepdft_runner.py',
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def reference_graph(atoms, probes, cutoff):
    combined = atoms.copy()
    combined.extend(Atoms(numbers=np.zeros(len(probes), dtype=int), positions=probes))
    src, dst, shifts = neighbor_list('ijS', combined, cutoff)
    keep = (src >= len(atoms)) & (dst < len(atoms))
    return np.stack((dst[keep], src[keep] - len(atoms)), axis=1), shifts[keep]


def rows(graph):
    # Sorting preserves multiplicity while allowing neighbor order to change.
    return sorted(map(tuple, np.column_stack(graph).tolist()))


@pytest.mark.parametrize('cell,pbc', [
    ([[2., 0, 0], [1.3, 1.7, 0], [.4, .6, 2.1]], [True] * 3),
    ([[2., 0, 0], [1.3, 1.7, 0], [.4, .6, 2.1]], [True, True, False]),
    (np.eye(3) * 5, [False] * 3),
    ([[2., 0, 0], [.8, 2., 0], [0, 0, 0]], [True, True, False]),
    (np.zeros((3, 3)), [False] * 3),
])
@pytest.mark.parametrize('unwrapped', [False, True])
def test_probe_graph_matches_ase(cell, pbc, unwrapped):
    rng = np.random.default_rng(42)
    atoms = Atoms('H3', positions=rng.random((3, 3)), cell=cell, pbc=pbc)
    probes = rng.random((7, 3)) @ atoms.cell.complete()
    if unwrapped:
        atoms.positions += rng.integers(-3, 4, (3, 3)) @ atoms.cell
        probes += rng.integers(-3, 4, (7, 3)) @ atoms.cell
    expected = reference_graph(atoms, probes, 3.1)
    actual = runner._probes_to_graph(atoms, probes, 3.1)
    assert rows(actual) == rows(expected)
    assert actual[0].dtype == np.int64
    assert actual[1].dtype == np.float32


@pytest.mark.parametrize('positions,probes,cutoff', [
    ([[0, 0, 0]], [], 1.),
    ([], [[0, 0, 0]], 1.),
    ([[0, 0, 0]], [[2, 0, 0]], 1.),
    ([[0, 0, 0]], [[1, 0, 0]], 1.),  # Strict cutoff boundary.
])
def test_probe_graph_empty(positions, probes, cutoff):
    atoms = Atoms(numbers=np.ones(len(positions), dtype=int), positions=positions)
    edges, shifts = runner._probes_to_graph(atoms, probes, cutoff)
    assert edges.shape == (0, 2)
    assert shifts.shape == (0, 3)


@pytest.mark.parametrize('budget', [2, 8])
def test_probe_graph_chunks_without_combined_neighbor_list(monkeypatch, budget):
    atoms = Atoms('H3', positions=[[0, 0, 0], [.2, .3, .4], [.5, .6, .7]],
                  cell=[2, 2, 2], pbc=True)
    probes = np.random.default_rng(12).random((11, 3)) * 2
    expected = reference_graph(atoms, probes, 1.5)
    monkeypatch.setattr(runner, '_MAX_PROBE_PAIRS', budget)

    def forbidden(*args, **kwargs):
        pytest.fail('Probe graph must not enumerate probe–probe pairs')

    monkeypatch.setattr(runner, 'neighbor_list', forbidden)
    original_einsum = np.einsum

    def bounded_einsum(expression, delta, other):
        assert delta.shape[0] * delta.shape[1] <= budget
        return original_einsum(expression, delta, other)

    monkeypatch.setattr(runner.np, 'einsum', bounded_einsum)
    assert rows(runner._probes_to_graph(atoms, probes, 1.5)) == rows(expected)


@pytest.mark.parametrize('legacy', [False, True])
@pytest.mark.parametrize('wrapped', [False, True])
@pytest.mark.parametrize('painn', [False, True])
def test_load_model_checkpoint(tmp_path, legacy, wrapped, painn):
    (tmp_path / 'arguments.json').write_text(json.dumps(dict(
        use_painn_model=painn, num_interactions=2, node_size=16, cutoff=4.,
    )))
    (tmp_path / 'best_model.pth').touch()
    parameters = {'weight': object()}
    checkpoint = {'model': parameters, 'args': argparse.Namespace()} if wrapped else parameters
    calls = []
    loaded = []
    model = SimpleNamespace(to=lambda device: loaded.append(device),
                            load_state_dict=lambda state: loaded.append(state),
                            eval=lambda: loaded.append('eval'))

    def factory(*args):
        assert args == (2, 16, 4.)
        return model

    def wrong_factory(*args):
        pytest.fail('Wrong architecture')

    def load(path, **kwargs):
        assert path == tmp_path / 'best_model.pth'
        assert kwargs['map_location'] == 'cpu'
        calls.append(kwargs)
        if legacy and 'weights_only' in kwargs:
            raise TypeError("unexpected keyword argument 'weights_only'")
        if not legacy:
            assert kwargs['weights_only'] is False
        return checkpoint

    modules = SimpleNamespace(PainnDensityModel=factory if painn else wrong_factory,
                              DensityModel=wrong_factory if painn else factory)
    torch = SimpleNamespace(device=lambda device: device, load=load)
    assert runner._load_model(tmp_path, 'cpu', modules, torch) == (model, 4.)
    assert loaded == ['cpu', parameters, 'eval']
    assert len(calls) == (2 if legacy else 1)
