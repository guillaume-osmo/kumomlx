"""kumomlx -- NVIDIA Kumo-Tabular on Apple Silicon via MLX.

    from kumomlx import KumoMLX, load_boiling_point

    data = load_boiling_point()
    (xc, yc), (xq, yq), _ = data.split(n_context=800)
    pred = KumoMLX().predict(xc, yc, xq)
"""

from kumomlx.datasets import BoilingPoint, load_boiling_point

__all__ = ["KumoMLX", "BoilingPoint", "load_boiling_point", "network", "weights", "__version__"]
__version__ = "0.1.0"


def __getattr__(name):
    # KumoMLX pulls in torch and the upstream recipe package; importing kumomlx should not.
    if name == "KumoMLX":
        from kumomlx.hybrid import KumoMLX
        return KumoMLX
    if name in ("network", "weights"):
        import importlib
        return importlib.import_module(f"kumomlx.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
