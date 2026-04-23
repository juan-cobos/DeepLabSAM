import pytest


def _parse(argv):
    import sys
    from deeplabsam.cli import main

    sys.argv = ["deeplabsam"] + argv
    with pytest.raises(SystemExit):
        main()


def test_image_subcommand_help(capsys):
    _parse(["image", "--help"])
    out = capsys.readouterr().out
    assert "--sam-model" in out
    assert "--pose-model" in out


def test_video_subcommand_help(capsys):
    _parse(["video", "--help"])
    out = capsys.readouterr().out
    assert "--sam-model" in out
    assert "--pose-model" in out


def test_unknown_pose_model_raises():
    import pytest
    from deeplabsam.models.dlc import DLCPose

    with pytest.raises(ValueError, match="Unknown model"):
        DLCPose(model="doesnotexist")


def test_registry_contains_topviewmouse():
    from deeplabsam.models.dlc import _REGISTRY

    assert "topviewmouse" in _REGISTRY
    assert _REGISTRY["topviewmouse"].endswith(".onnx")
