from pathlib import Path
import sys

# --- chemins -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))          # rend import taxiV3 possible

# --- projet ------------------------------------------------------------------
project = "TaxiV3"
author = "Florian Torchy"
version = release = "0.1.0"

# --- extensions --------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
]
autosummary_generate = True
autodoc_typehints = "description"

# --- thème -------------------------------------------------------------------
html_theme = "sphinx_rtd_theme"

# --- mock d’imports lourds ---------------------------------------------------
autodoc_mock_imports = ["torch", "gymnasium"]