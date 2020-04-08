import nbformat
import nbconvert
from pathlib import Path

if __name__ == "__main__":
    file_dir = Path(__file__).absolute().parent
    nb_paths = sorted((file_dir / "Basic").glob("*.ipynb"))
    nb_paths += sorted((file_dir / "Advanced").glob("*.ipynb"))
    for nb_path in nb_paths:
        print(f"[Executing notebook {nb_path.name}]")

        with open(nb_path) as f:
            nb = nbformat.read(f, as_version=4)
        ep = nbconvert.preprocessors.ExecutePreprocessor(timeout=6000)
        try:
            ep.preprocess(nb, {"metadata": {"path": nb_path.parent}})
        except nbconvert.preprocessors.execute.CellExecutionError:
            print(f"Execution of {nb_path.name} failed")

        with open(nb_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
