import nbformat
import nbconvert
from pathlib import Path
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action='store_true')
    parser.add_argument("--break_on_failure", action='store_true')
    args = parser.parse_args()

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

    print("Found the following notebooks:")
    for nb_path in nb_paths:
        print("> {}".format(nb_path))

    for nb_path in nb_paths:
        print("[Executing notebook {}]".format(nb_path.name))

        with open(nb_path) as f:
            nb = nbformat.read(f, as_version=4)
        ep = nbconvert.preprocessors.ExecutePreprocessor(timeout=6000)
        try:
            ep.preprocess(nb, {"metadata": {"path": nb_path.parent}})
        except nbconvert.preprocessors.execute.CellExecutionError as e:
            print("Execution of {} failed".format(nb_path.name))
            if args.break_on_failure:
                raise

        if args.write:
            print("Writing the executed notebook")
            with open(nb_path, "w", encoding="utf-8") as f:
                nbformat.write(nb, f)
