import os
import pytest

# Prevent OpenBLAS/MKL/OpenMP threading deadlocks on Windows (Python 3.14 + sklearn)
# Must be set BEFORE any scientific library is imported.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')


# Phase K.4 (2026-05-14) — auto-restore DASHBOARD_API_KEY after every test.
# Test pollution: test_dashboard.py contains lines like
#   os.environ['DASHBOARD_API_KEY'] = ''
# inside individual tests with no setUp/tearDown to restore. When those run
# before test_dashboard_api.py, the empty key persists, the require_api_key
# decorator treats "" as configured-but-empty, all probe requests get 401,
# and ~30 tests fail. This autouse fixture snapshots the env value before
# each test and restores it after, so accidental mutations don't leak.
@pytest.fixture(autouse=True)
def _restore_dashboard_api_key_after_test():
    original = os.environ.get('DASHBOARD_API_KEY')
    try:
        yield
    finally:
        current = os.environ.get('DASHBOARD_API_KEY')
        if current != original:
            if original is None:
                os.environ.pop('DASHBOARD_API_KEY', None)
            else:
                os.environ['DASHBOARD_API_KEY'] = original


@pytest.fixture
def base_url():
    return 'http://127.0.0.1:5000'
