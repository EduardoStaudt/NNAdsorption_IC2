"""TensorFlow runtime setup. Import this module BEFORE tensorflow is imported anywhere.

Two machine-specific fixes live here:
- XLA compiles GPU code with `ptxas`. On WSL2 the system ptxas (CUDA 12.3) does not know
  the RTX 50xx (compute capability 12.0), so we point XLA to the newer ptxas shipped in the
  pip package `nvidia-cuda-nvcc-cu12`.
- TensorFloat-32 is on by default for float32 matmuls on recent GPUs. It keeps only ~10
  mantissa bits (relative error ~1e-3), which spoils the second derivatives of the PINN, so
  `setup_tf` turns it off unless the config asks for it.
Run-to-run determinism (`runtime.deterministic`): every random draw of the package is seeded
(numpy generators, Sobol scrambles, Keras initializers through `runtime.seed`), but GPU kernels
may still add floating-point numbers in a different order from run to run; two runs of the same
config then drift apart after ~1000 steps (Phase 3: y R2 0.927 vs 0.903). With the option on,
TensorFlow (and XLA) only use deterministic GPU algorithms.
GPU memory: by default a run takes memory as it needs it. With `runtime.gpu_memory_mb` it gets a
fixed amount instead, so that several runs can share one GPU without taking memory from each other.
"""

import importlib.util
import os


def _point_xla_to_pip_ptxas():
    if "--xla_gpu_cuda_data_dir" in os.environ.get("XLA_FLAGS", ""):
        return
    try:
        spec = importlib.util.find_spec("nvidia.cuda_nvcc")
    except ModuleNotFoundError:
        return
    if spec is None or not spec.submodule_search_locations:
        return
    root = list(spec.submodule_search_locations)[0]
    bin_dir = os.path.join(root, "bin")
    if os.path.exists(os.path.join(bin_dir, "ptxas")):
        flags = os.environ.get("XLA_FLAGS", "")
        os.environ["XLA_FLAGS"] = (flags + " --xla_gpu_cuda_data_dir=" + root).strip()
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")


_point_xla_to_pip_ptxas()
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf  # noqa: E402  (must come after the environment setup above)


def set_float_dtype(dtype):
    """Make new Keras layers compute in `dtype` ('float32' or 'float64').

    Keras 3 caches its global dtype policy the first time a layer is built, so changing only
    floatx is not enough: the policy must be set too.
    """
    tf.keras.backend.set_floatx(dtype)
    tf.keras.config.set_dtype_policy(dtype)


def setup_tf(runtime_cfg):
    """Apply the `runtime:` section of the config. Returns the float dtype name."""
    tf.config.experimental.enable_tensor_float_32_execution(bool(runtime_cfg["tf32"]))
    limit_mb = runtime_cfg.get("gpu_memory_mb")
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            if limit_mb:  # a fixed share of the GPU, so that several runs fit on it together
                tf.config.set_logical_device_configuration(
                    gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=int(limit_mb))])
            else:  # take GPU memory as it is needed
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass  # GPU already initialised (e.g. inside pytest); memory settings cannot change anymore
    dtype = runtime_cfg["dtype"]
    set_float_dtype(dtype)
    tf.keras.utils.set_random_seed(int(runtime_cfg["seed"]))  # Python, NumPy and TensorFlow seeds
    if runtime_cfg.get("deterministic"):
        tf.config.experimental.enable_op_determinism()  # same results run after run (GPU included)
    return dtype


def memory_peaks():
    """Peak memory of this process so far, in MB: GPU memory allocated by TensorFlow (None without a
    GPU) and resident host memory (None where the `resource` module is missing)."""
    gpu = tf.config.experimental.get_memory_info("GPU:0")["peak"] / 2**20 if tf.config.list_physical_devices("GPU") else None
    try:
        import resource

        host = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux: kB
    except ImportError:
        host = None
    return dict(gpu_peak_mb=gpu, host_peak_rss_mb=host)


def device_name():
    """'GPU' when TensorFlow sees a GPU, else 'CPU' (used to report GPU hours honestly)."""
    return "GPU" if tf.config.list_physical_devices("GPU") else "CPU"
