"""Sphinx configuration for Ethnicolr."""

from py_canon.sphinx import configure

configure(
    globals(),
    master_doc="index",
    language="en",
    html_theme_options={
        "light_css_variables": {
            "color-brand-primary": "#336790",
            "color-brand-content": "#336790",
        },
        "dark_css_variables": {
            "color-brand-primary": "#4db8ff",
            "color-brand-content": "#4db8ff",
        },
        "navigation_with_keys": True,
        "sidebar_hide_name": False,
        "top_of_page_buttons": ["view", "edit"],
    },
    html_static_path=["_static"],
)

extensions.append("sphinx_autodoc_typehints")
intersphinx_mapping.update(
    {
        "numpy": ("https://numpy.org/doc/stable", None),
        "pandas": ("https://pandas.pydata.org/docs", None),
        "sklearn": ("https://scikit-learn.org/stable", None),
    }
)

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}
