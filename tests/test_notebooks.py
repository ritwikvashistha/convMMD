"""Structure and executable smoke checks for the canonical notebooks."""

import ast
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = PROJECT_ROOT / "examples" / "notebooks"
DECONV_NOTEBOOK = NOTEBOOK_DIR / "deconvolution_density_estimation.ipynb"
DENOISING_NOTEBOOK = NOTEBOOK_DIR / "posterior_mean_denoising.ipynb"
REGRESSION_NOTEBOOK = NOTEBOOK_DIR / "linear_measurement_error_regression.ipynb"
NOTEBOOKS = (DECONV_NOTEBOOK, DENOISING_NOTEBOOK, REGRESSION_NOTEBOOK)
SCRIPT_EXAMPLES = (
    PROJECT_ROOT / "examples" / "deconv_1d.py",
    PROJECT_ROOT / "examples" / "deconv_2d.py",
)


def _cell_source(cell):
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


def _notebook(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _code_cells(path):
    return [
        _cell_source(cell)
        for cell in _notebook(path)["cells"]
        if cell.get("cell_type") == "code"
    ]


def test_exact_canonical_notebook_set():
    assert set(NOTEBOOK_DIR.glob("*.ipynb")) == set(NOTEBOOKS)


@pytest.mark.parametrize("notebook_path", NOTEBOOKS, ids=lambda path: path.stem)
def test_notebook_json_python_cells_and_clean_outputs(notebook_path):
    notebook = _notebook(notebook_path)

    assert notebook["nbformat"] == 4
    assert notebook["cells"]
    for index, cell in enumerate(notebook["cells"]):
        assert cell["cell_type"] in {"code", "markdown"}
        if cell["cell_type"] == "code":
            source = _cell_source(cell)
            compile(source, f"{notebook_path.name}:cell-{index}", "exec")
            assert cell.get("execution_count") is None
            assert cell.get("outputs", []) == []


@pytest.mark.parametrize("notebook_path", NOTEBOOKS, ids=lambda path: path.stem)
def test_notebooks_do_not_reimplement_package_methods(notebook_path):
    definitions = []
    for source in _code_cells(notebook_path):
        tree = ast.parse(source)
        definitions.extend(
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        )

    assert definitions == []


def test_deconvolution_notebook_uses_public_package_training_api():
    source = "\n".join(_code_cells(DECONV_NOTEBOOK))
    fit_cell = next(
        cell for cell in _code_cells(DECONV_NOTEBOOK) if "fit = train_convmmd" in cell
    )

    assert "from convMMD.density_models import NormalizingFlowDensity" in source
    assert "from convMMD.training import train_convmmd" in source
    assert "x_noisy=noisy_observations" in source
    assert "latent_truth" not in fit_cell


def test_denoising_notebook_uses_default_importance_sampling():
    source = "\n".join(_code_cells(DENOISING_NOTEBOOK))
    fit_cell = next(
        cell for cell in _code_cells(DENOISING_NOTEBOOK) if "result = denoise" in cell
    )

    assert "from convMMD import denoise" in source
    assert "posterior_method=" not in fit_cell
    assert 'result.config.posterior_method == "importance"' in fit_cell
    assert "latent_truth" not in fit_cell


def test_regression_notebook_cannot_pass_truth_to_fitting():
    source = "\n".join(_code_cells(REGRESSION_NOTEBOOK))
    fit_cell = next(
        cell
        for cell in _code_cells(REGRESSION_NOTEBOOK)
        if "result = fit_measurement_error_regression" in cell
    )

    assert "from convMMD import fit_measurement_error_regression" in source
    assert "observed_covariate" in fit_cell
    assert "response" in fit_cell
    assert "latent_truth" not in fit_cell
    assert "TRUE_" not in fit_cell


@pytest.mark.parametrize("script_path", SCRIPT_EXAMPLES, ids=lambda path: path.stem)
def test_deconvolution_scripts_keep_truth_out_of_training_calls(script_path):
    tree = ast.parse(script_path.read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "train_convmmd"
    ]

    assert len(calls) == 1
    call = calls[0]
    assert "eval_fn" not in {keyword.arg for keyword in call.keywords}
    assert "theta_true" not in {
        node.id for node in ast.walk(call) if isinstance(node, ast.Name)
    }


@pytest.mark.parametrize("notebook_path", NOTEBOOKS, ids=lambda path: path.stem)
def test_notebook_reduced_configuration_executes(notebook_path, monkeypatch):
    monkeypatch.setenv("CONVMMD_NOTEBOOK_SMOKE", "1")
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.chdir(PROJECT_ROOT)
    monkeypatch.setattr(plt, "show", lambda: None)
    namespace = {"__name__": "__main__"}

    try:
        for index, source in enumerate(_code_cells(notebook_path)):
            exec(
                compile(source, f"{notebook_path.name}:cell-{index}", "exec"),
                namespace,
            )
    finally:
        plt.close("all")
