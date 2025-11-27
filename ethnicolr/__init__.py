"""
ethnicolr: Predict Race/Ethnicity Based on Names

A Python package for predicting race and ethnicity based on names using
machine learning models trained on Census, Wikipedia, and voter registration data.
"""

from __future__ import annotations

__all__ = [
    "census_ln",
    "pred_census_ln",
    "pred_wiki_ln",
    "pred_wiki_name",
    "pred_fl_reg_ln",
    "pred_fl_reg_name",
    "pred_nc_reg_name",
    "pred_fl_reg_ln_five_cat",
    "pred_fl_reg_name_five_cat",
]

# Lazy imports to avoid loading TensorFlow until needed
_LAZY_IMPORTS = {
    "census_ln": ("ethnicolr.census_ln", "census_ln"),
    "pred_census_ln": ("ethnicolr.pred_census_ln", "pred_census_ln"),
    "pred_fl_reg_ln": ("ethnicolr.pred_fl_reg_ln", "pred_fl_reg_ln"),
    "pred_fl_reg_ln_five_cat": (
        "ethnicolr.pred_fl_reg_ln_five_cat",
        "pred_fl_reg_ln_five_cat",
    ),
    "pred_fl_reg_name": ("ethnicolr.pred_fl_reg_name", "pred_fl_reg_name"),
    "pred_fl_reg_name_five_cat": (
        "ethnicolr.pred_fl_reg_name_five_cat",
        "pred_fl_reg_name_five_cat",
    ),
    "pred_nc_reg_name": ("ethnicolr.pred_nc_reg_name", "pred_nc_reg_name"),
    "pred_wiki_ln": ("ethnicolr.pred_wiki_ln", "pred_wiki_ln"),
    "pred_wiki_name": ("ethnicolr.pred_wiki_name", "pred_wiki_name"),
}


def __getattr__(name: str):
    """Lazy import implementation."""
    if name in _LAZY_IMPORTS:
        module_name, func_name = _LAZY_IMPORTS[name]
        import importlib

        module = importlib.import_module(module_name)
        func = getattr(module, func_name)
        # Cache the function in the module's globals for future access
        globals()[name] = func
        return func
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
