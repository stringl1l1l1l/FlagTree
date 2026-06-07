# flagtree backend path specialization
from triton.flagtree_spec import spec_path

spec_path(__path__)

from . import hopper
from . import blackwell

__all__ = ["hopper", "blackwell"]
