from jax.tree_util import PyTreeDef
import pickle
from pathlib import Path
from typing import Union

SUFFIX_PICKLE = '.pickle'


def save_pytree(data: PyTreeDef,
                path: Union[str, Path],
                overwrite: bool = True
                ):
    path = Path(path)
    if path.suffix != SUFFIX_PICKLE:
        path = path.with_suffix(SUFFIX_PICKLE)
    path.parent.mkdir(parents=True,
                      exist_ok=True)
    if path.exists():
        if overwrite:
            path.unlink()
        else:
            raise RuntimeError(f'File {path} already exists.')
    with open(path,
              'wb') as file:
        pickle.dump(data,
                    file,
                    protocol=pickle.HIGHEST_PROTOCOL)


def load_pytree(path: Union[str, Path]) -> PyTreeDef:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Not a file: {path}')
    if path.suffix != SUFFIX_PICKLE:
        raise ValueError(f'Not a {SUFFIX_PICKLE} file: {path}')
    with open(path,
              'rb') as file:
        data = pickle.load(file)
    return data
