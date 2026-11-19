import subprocess
import sys

import pytest

from conftest import REPO_ROOT
from nnads_ic2 import runtime

# The GPU of the test process is already initialised (memory settings are fixed by then), so the
# memory limit is checked in a fresh process.
LIMIT_CHECK = r'''
import sys
sys.path.insert(0, "src")
from nnads_ic2 import runtime
import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
if not gpus:
    print("no GPU")
    sys.exit(0)
runtime.setup_tf(dict(tf32=False, dtype="float32", seed=0, deterministic=False, gpu_memory_mb=256))
print("limit", tf.config.get_logical_device_configuration(gpus[0])[0].memory_limit)
try:
    x = tf.ones((1024, 1024, 128))  # 512 MB, twice the limit
    print("allocated", float(tf.reduce_sum(x)))
except tf.errors.ResourceExhaustedError:
    print("out of memory")
'''


def test_gpu_memory_limit_caps_what_tensorflow_may_allocate():
    out = subprocess.run([sys.executable, "-c", LIMIT_CHECK], cwd=REPO_ROOT, capture_output=True, text=True, timeout=300).stdout
    if "no GPU" in out:
        pytest.skip("no GPU")
    assert "limit 256" in out and "out of memory" in out


def test_memory_peaks_are_reported():
    peaks = runtime.memory_peaks()
    assert peaks["host_peak_rss_mb"] > 100  # TensorFlow alone takes more than that
    assert peaks["gpu_peak_mb"] is None or peaks["gpu_peak_mb"] >= 0
