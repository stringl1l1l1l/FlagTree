# flagtree backend path specialization
from triton.flagtree_spec import spec_path

spec_path(__path__)

from . import gfx1250

__all__ = ["gfx1250"]
