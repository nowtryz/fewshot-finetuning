import pytest
import torch


def match_keyword(metafunc: pytest.Metafunc, keyword: str):
    return keyword in metafunc.definition.originalname or any(keyword in name for name in metafunc.fixturenames)


def pytest_generate_tests(metafunc: pytest.Metafunc):
    """Automatic test marking and fixture auto-use"""
    if match_keyword(metafunc, 'rand'):  # Any test using randomness are executed multiple times unless stated otherwise
        metafunc.definition.add_marker(pytest.mark.repeat(3))


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item: pytest.Function):
    # Any test mentioning backward are wrapped with torch.autograd.detect_anomaly
    if 'backward' in item.name or 'grad' in item.name:
        with torch.autograd.detect_anomaly():
            yield
    else:
        yield
