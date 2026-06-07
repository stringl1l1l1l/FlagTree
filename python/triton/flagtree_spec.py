import importlib
import os
import sys

_spec_module = None


def _triton_root() -> str | None:
    current_path = os.path.abspath(__file__).replace(os.sep, "/")
    marker = "/triton"
    idx = current_path.find(marker)
    if idx == -1:
        return None
    return current_path[:idx + len(marker)]


def _get_spec_module():
    global _spec_module
    from ._flagtree_backend import FLAGTREE_BACKEND
    if not FLAGTREE_BACKEND:
        return None
    triton_root = _triton_root()
    if triton_root is None:
        return None
    spec_dir = os.path.join(triton_root, "backends", FLAGTREE_BACKEND, "spec")
    if not os.path.isdir(spec_dir):
        return None
    if _spec_module is not None:
        return _spec_module
    try:
        _spec_module = importlib.import_module(f"triton.backends.{FLAGTREE_BACKEND}.spec")
    except ImportError:
        return None
    return _spec_module


# flagtree backend path specialization
def spec_path(path_list: list, exclude=()):
    from ._flagtree_backend import FLAGTREE_BACKEND
    if not path_list or not FLAGTREE_BACKEND:
        return
    current_path = path_list[0].replace(os.sep, "/")
    marker = "/triton"
    idx = current_path.find(marker)
    if idx == -1:
        return
    triton_root = current_path[:idx + len(marker)]
    rel_path = current_path[idx + 1 + len(marker):]

    # lookup backends/<backend>/spec/triton/<rel_path>
    backend_path = os.path.join(triton_root, "backends", FLAGTREE_BACKEND, "spec", "triton", rel_path)
    if os.path.isdir(backend_path) and backend_path not in path_list:
        path_list.insert(0, backend_path)

    # lookup third_party/<backend>/python/triton/<rel_path> (for editable installs)
    project_root = os.path.dirname(os.path.dirname(triton_root))
    third_party_path = os.path.join(project_root, "third_party", FLAGTREE_BACKEND, "python", "triton", rel_path)
    if os.path.isdir(third_party_path) and third_party_path not in path_list:
        if exclude:
            _protect_subpackages(triton_root, exclude)
        path_list.insert(0, third_party_path)


# flagtree backend specialization
def spec(function_name: str, *args, **kwargs):
    mod = _get_spec_module()
    if mod is not None and hasattr(mod, function_name):
        return getattr(mod, function_name)(*args, **kwargs)
    return None


# flagtree backend func specialization
def spec_func(function_name: str):
    mod = _get_spec_module()
    if mod is not None and hasattr(mod, function_name):
        return getattr(mod, function_name)
    return None


# flagtree language extension
def bind_language_extension_symbols_to_tl(extension):
    import triton.language as tl

    names = getattr(extension, "__all__", None)
    if not names:
        return

    for name in names:
        if not hasattr(extension, name):
            continue
        if hasattr(tl, name):
            continue
        setattr(tl, name, getattr(extension, name))


class _OriginalPathFinder:
    """Forces certain subpackages to load from the original triton path,
    bypassing any overlay in __path__."""

    def __init__(self, original_triton_root, names):
        self._original = original_triton_root
        self._protected = {f"triton.{n}" for n in names}

    def find_spec(self, fullname, path, target=None):
        if fullname not in self._protected:
            return None
        name = fullname.split(".")[-1]
        pkg_dir = os.path.join(self._original, name)
        if not os.path.isdir(pkg_dir):
            return None
        import importlib.util
        init = os.path.join(pkg_dir, "__init__.py")
        if not os.path.isfile(init):
            return None
        return importlib.util.spec_from_file_location(fullname, init, submodule_search_locations=[pkg_dir])


def _protect_subpackages(original_triton_root, names):
    for finder in sys.meta_path:
        if isinstance(finder, _OriginalPathFinder):
            return
    sys.meta_path.insert(0, _OriginalPathFinder(original_triton_root, names))
