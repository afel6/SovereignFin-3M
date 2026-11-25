import pytest


@pytest.fixture
def allowed():
    return frozenset({"vendors", "payments"})
