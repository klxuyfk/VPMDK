"""Export publication is conditional on upstream numerical verification."""
import json
from pathlib import Path
import zipfile

import pytest

import vpmdk  # Select the checkout through the legacy compatibility entry point.

from vpmdk_core import mace_soyo_export as export


def _inputs(tmp_path):
    checkpoint = tmp_path / "model.pth"
    structure = tmp_path / "POSCAR"
    checkpoint.write_bytes(b"weights")
    structure.write_text("structure")
    output = tmp_path / "model.pt2"
    return ["--ckpt-path", str(checkpoint), "--structure-file", str(structure),
            "--output-path", str(output)], output


def _package(arguments):
    path = Path(arguments[arguments.index("--output-path") + 1])
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("model/data/aotinductor/model/model.wrapper_metadata.json",
            json.dumps({"mace_soyo_aoti_outputs": "energy forces stress",
                        "mace_soyo_aoti_head_names": '["mptraj"]'}))


def test_verified_export_is_published_with_provenance(tmp_path, monkeypatch):
    arguments, output = _inputs(tmp_path)
    calls = []
    def run(args):
        calls.append(args)
        assert not output.exists()
        _package(args)
    monkeypatch.setattr(export, "_run_upstream_export", run)
    monkeypatch.setattr(export, "_runtime_info", lambda device: {"gpu": "test GPU"})
    export.main([*arguments, "--max-nodes", "128", "--max-edges", "4096",
                 "--max-graphs", "8", "--inductor-config", "max_autotune=false"])
    assert output.is_file()
    report = json.loads(output.with_suffix(".pt2.json").read_text())
    assert report["verified"] is True
    assert report["package_sha256"] == export._sha256(output)
    assert report["checkpoint_sha256"] == export._sha256(tmp_path / "model.pth")
    assert "--max-nodes" in calls[0] and "max_autotune=false" in calls[0]
    assert not list(tmp_path.glob(".model-*"))


def test_failed_consistency_check_preserves_previous_package(tmp_path, monkeypatch):
    arguments, output = _inputs(tmp_path)
    output.write_bytes(b"old verified package")
    report = output.with_suffix(".pt2.json")
    report.write_text("old report")
    def fail(args):
        _package(args)
        raise AssertionError("forces do not match")
    monkeypatch.setattr(export, "_run_upstream_export", fail)
    with pytest.raises(AssertionError, match="forces do not match"):
        export.main([*arguments, "--overwrite"])
    assert output.read_bytes() == b"old verified package"
    assert report.read_text() == "old report"
    assert not list(tmp_path.glob(".model-*"))


def test_existing_output_requires_explicit_overwrite(tmp_path):
    arguments, output = _inputs(tmp_path)
    output.touch()
    with pytest.raises(SystemExit):
        export.main(arguments)


@pytest.mark.parametrize("option", [["--device", "cpu"], ["--max-graphs", "1"],
                                   ["--max-nodes", "1"], ["--max-edges", "1"]])
def test_export_validation_precedes_optional_imports(tmp_path, option, monkeypatch):
    arguments, _ = _inputs(tmp_path)
    monkeypatch.setattr(export, "_run_upstream_export", lambda args: pytest.fail("export ran"))
    with pytest.raises(SystemExit):
        export.main([*arguments, *option])


def test_package_metadata_requires_efs(tmp_path, monkeypatch):
    arguments, output = _inputs(tmp_path)
    def invalid(args):
        path = Path(args[args.index("--output-path") + 1])
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("test.wrapper_metadata.json", '{}')
    monkeypatch.setattr(export, "_run_upstream_export", invalid)
    with pytest.raises(ValueError, match="energy/forces/stress"):
        export.main(arguments)
    assert not output.exists()


def test_upstream_argv_restored_on_error(monkeypatch):
    import sys
    from types import ModuleType
    module = ModuleType("mace_soyo.export.AOTI_export")
    def fail():
        assert sys.argv == ["mace-soyo-aoti-export", "--device", "cuda:0"]
        raise RuntimeError("export failed")
    module.main = fail
    monkeypatch.setitem(sys.modules, "mace_soyo.export.AOTI_export", module)
    monkeypatch.setattr(export, "_verify_native_ops", lambda device: None)
    from contextlib import nullcontext
    from vpmdk_core.backends import mace_soyo_common
    monkeypatch.setattr(mace_soyo_common, "_cueq_checkpoint_compat", nullcontext)
    previous = sys.argv
    with pytest.raises(RuntimeError, match="export failed"):
        export._run_upstream_export(["--device", "cuda:0"])
    assert sys.argv is previous


def test_report_cannot_replace_input_structure(tmp_path, monkeypatch):
    arguments, output = _inputs(tmp_path)
    structure = output.with_suffix(".pt2.json")
    from ase.build import bulk
    from ase.io import write
    write(structure, bulk("Si", "diamond", a=5.43), format="json")
    original = structure.read_bytes()
    arguments[arguments.index("--structure-file") + 1] = str(structure)
    monkeypatch.setattr(export, "_run_upstream_export", _package)
    monkeypatch.setattr(export, "_runtime_info", lambda device: {})
    with pytest.raises(SystemExit):
        export.main([*arguments, "--overwrite"])
    assert structure.read_bytes() == original
    assert not output.exists()



@pytest.mark.parametrize("input_tag", ["--ckpt-path", "--structure-file"])
@pytest.mark.parametrize("destination", ["package", "report"])
@pytest.mark.parametrize("alias", ["direct", "input_symlink", "parent_symlink"])
def test_all_export_destinations_are_distinct_from_resolved_inputs(
    tmp_path, monkeypatch, input_tag, destination, alias
):
    arguments, output = _inputs(tmp_path)
    target = output if destination == "package" else output.with_suffix(".pt2.json")
    target.write_bytes(b"input bytes must survive")
    input_path = target
    if alias == "input_symlink":
        input_path = tmp_path / "input-link"
        input_path.symlink_to(target)
    elif alias == "parent_symlink":
        parent = tmp_path / "parent-link"
        parent.symlink_to(tmp_path, target_is_directory=True)
        output_alias = parent / output.name
        arguments[arguments.index("--output-path") + 1] = str(output_alias)
    arguments[arguments.index(input_tag) + 1] = str(input_path)
    originals = {path: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()}
    monkeypatch.setattr(export, "_run_upstream_export", lambda args: pytest.fail("export started"))
    monkeypatch.setattr(export, "_runtime_info", lambda device: pytest.fail("GPU queried"))
    with pytest.raises(SystemExit):
        export.main([*arguments, "--overwrite"])
    for path, data in originals.items():
        assert path.read_bytes() == data
    assert not list(tmp_path.glob(".model-*"))
