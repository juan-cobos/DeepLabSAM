from pathlib import Path

import cv2
import pytest

EXAMPLE_IMAGE = Path(__file__).resolve().parents[1] / "examples" / "images" / "example.png"
EXAMPLE_BOX = [607, 450, 770, 726]


@pytest.fixture(scope="session")
def example_image():
    if not EXAMPLE_IMAGE.exists():
        pytest.skip(f"Example image missing: {EXAMPLE_IMAGE}")
    image = cv2.imread(str(EXAMPLE_IMAGE))
    if image is None:
        pytest.skip(f"Cannot read: {EXAMPLE_IMAGE}")
    return image


@pytest.fixture(scope="session")
def example_box():
    return EXAMPLE_BOX
