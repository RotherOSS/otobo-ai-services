from pathlib import Path
import importlib.util


def relative_import(module_name: str, file: str = __file__):
    """
    Use this to import modules within this rag module to keep naming dynamic.
    Importantly, pass over __file__ from where it is called

    Example:
        rag_chain = relative_import("chains", file=__file__).rag_chain
    """

    module_path = Path(file).resolve().parent / f"{module_name}.py"

    if not module_path.exists():
        raise ModuleNotFoundError(
            f"Could not find module '{module_name}' at: {module_path}"
        )

    spec = importlib.util.spec_from_file_location(module_name, module_path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for module '{module_name}'")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module