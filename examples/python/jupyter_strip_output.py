import nbformat
import nbconvert
from pathlib import Path
import os

if __name__ == "__main__":
    # Setting os.environ["CI"] will disable interactive (blocking) mode in
    # Jupyter notebooks
    os.environ["CI"] = "true"

    file_dir = Path(__file__).absolute().parent

    # Note: must be consistent with make_docs.py
    example_dirs = [
        "geometry",
        "core",
        "pipelines",
        "visualization",
    ]
    nb_paths = []
    for example_dir in example_dirs:
        nb_paths += sorted((file_dir / example_dir).glob("*.ipynb"))

    for nb_path in nb_paths:
        print("Clean {}".format(nb_path.name))

        with open(nb_path) as f:
            nb = nbformat.read(f, as_version=4)
        ep = nbconvert.preprocessors.ClearOutputPreprocessor(timeout=6000)
        try:
            ep.preprocess(nb, {"metadata": {"path": nb_path.parent}})
        except nbconvert.preprocessors.execute.CellExecutionError:
            print("Cleaning of {} failed".format(nb_path.name))

        with open(nb_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
