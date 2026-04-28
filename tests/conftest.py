import os
import pytest

# Prevent OpenBLAS/MKL/OpenMP threading deadlocks on Windows (Python 3.14 + sklearn)
# Must be set BEFORE any scientific library is imported.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')


@pytest.fixture
def base_url():
    return 'http://127.0.0.1:5000'
