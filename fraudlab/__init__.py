"""fraudlab: reproducible research utilities for the fraud-detection project.

Everything here is CPU-friendly. TensorFlow is imported lazily inside the
modules that need it so that data, metrics, drift and audit code can be used
(and tested) without it.
"""
__version__ = "0.2.0"
