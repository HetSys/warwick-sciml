"""Wrapped tqdm/trange with sensible defaults for RISE and Quarto.

Avoids IOPub message rate limiting in Jupyter Notebook classic by
overriding set_description/set_postfix_str to not force a refresh
on every call. Updates appear on the next natural refresh cycle,
controlled by mininterval.
"""

import os

from tqdm import tqdm as _tqdm

# Detect Quarto rendering mode (disable progress bars in static output)
QUARTO_MODE = os.environ.get('QUARTO_PROJECT_DIR') is not None
SHOW_PROGRESS = not QUARTO_MODE

# numpyro updates its progress bar from a JAX callback thread. marimo (>= 0.25)
# rejects stdout writes from threads with no active cell, which makes MCMC hang,
# so pass NUMPYRO_PROGRESS (not SHOW_PROGRESS) as numpyro's progress_bar.
try:
    import marimo as _mo

    _IN_MARIMO = _mo.running_in_notebook()
except ImportError:
    _IN_MARIMO = False
NUMPYRO_PROGRESS = SHOW_PROGRESS and not _IN_MARIMO


class tqdm(_tqdm):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('leave', False)
        kwargs.setdefault('mininterval', 0.5)
        kwargs.setdefault('disable', not SHOW_PROGRESS)
        super().__init__(*args, **kwargs)

    def set_description(self, desc=None, refresh=False):
        super().set_description(desc, refresh=refresh)

    def set_postfix_str(self, s='', refresh=False):
        super().set_postfix_str(s, refresh=refresh)

    def set_postfix(self, ordered_dict=None, refresh=False, **kwargs):
        super().set_postfix(ordered_dict, refresh=refresh, **kwargs)


def trange(*args, **kwargs):
    return tqdm(range(*args), **kwargs)
