from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio

from kinship.paths import resolve_user_path, workspace_root


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _normalize_columns(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _fit_projection(data: np.ndarray, n_components: int, rng: np.random.Generator) -> np.ndarray:
    centered = data - data.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[: min(n_components, vt.shape[0])].T
    if basis.shape[1] < n_components:
        padding = rng.normal(scale=1e-2, size=(data.shape[1], n_components - basis.shape[1]))
        basis = np.concatenate([basis, padding], axis=1)
    return _normalize_columns(basis[:, :n_components].astype(np.float64))


def _fit_mapping(interaction: np.ndarray, nummap: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    basis = _fit_projection(interaction, nummap, rng).T
    bias = -(interaction.mean(axis=0) @ basis.T)
    return basis.astype(np.float64), bias.astype(np.float64)


def _reconstruction_error(
    x: np.ndarray,
    y: np.ndarray,
    wxf: np.ndarray,
    wyf: np.ndarray,
    whf: np.ndarray,
    bmap: np.ndarray,
    wpf: np.ndarray | None = None,
) -> float:
    fx = x @ wxf
    fy = y @ wyf
    interaction = fx * fy
    if wpf is not None:
        encoded = _softplus(interaction) @ wpf.T
        factors_h = (_sigmoid(encoded @ whf.T + bmap) @ whf) @ wpf
    else:
        factors_h = _sigmoid(interaction @ whf.T + bmap) @ whf
    x_hat = (fy * factors_h) @ wxf.T
    y_hat = (fx * factors_h) @ wyf.T
    return float(0.5 * np.mean((x - x_hat) ** 2) + 0.5 * np.mean((y - y_hat) ** 2))


def _solve_decoder(design: np.ndarray, target: np.ndarray) -> np.ndarray:
    solution, *_ = np.linalg.lstsq(design, target, rcond=None)
    return _normalize_columns(solution.T.astype(np.float64))


def _standard_gae(
    x: np.ndarray,
    y: np.ndarray,
    numfac: int,
    nummap: int,
    learnrate: float,
    numepochs: int,
    rng: np.random.Generator,
    verbose: bool,
) -> tuple[dict[str, np.ndarray], list[float]]:
    wxf = _fit_projection(x, numfac, rng)
    wyf = _fit_projection(y, numfac, rng)
    whf = np.zeros((nummap, numfac), dtype=np.float64)
    bmap = np.zeros(nummap, dtype=np.float64)
    blend = float(np.clip(learnrate * 25.0, 0.05, 0.35))
    history: list[float] = []
    for epoch in range(max(1, numepochs)):
        fx = x @ wxf
        fy = y @ wyf
        interaction = fx * fy
        whf, bmap = _fit_mapping(interaction, nummap, rng)
        gates = _sigmoid(interaction @ whf.T + bmap)
        factors_h = gates @ whf
        new_wxf = _solve_decoder(fy * factors_h, x)
        new_wyf = _solve_decoder(fx * factors_h, y)
        wxf = (1.0 - blend) * wxf + blend * new_wxf
        wyf = (1.0 - blend) * wyf + blend * new_wyf
        error = _reconstruction_error(x, y, wxf, wyf, whf, bmap)
        history.append(error)
        if verbose and (epoch == 0 or (epoch + 1) % 10 == 0 or epoch == numepochs - 1):
            print(f"epoch {epoch + 1}: reconstruction_error={error:.6f}")
    return {"wxf": wxf, "wyf": wyf, "whf": whf, "z_bias": bmap}, history


def _build_pool_matrix(numfac: int, subspace_dims: int) -> np.ndarray:
    numpool = int(np.ceil(numfac / subspace_dims))
    wpf = np.zeros((numpool, numfac), dtype=np.float64)
    for pool_index in range(numpool):
        start = pool_index * subspace_dims
        stop = min(start + subspace_dims, numfac)
        wpf[pool_index, start:stop] = 1.0
    return wpf


def _multiview_gae(
    x: np.ndarray,
    y: np.ndarray,
    numfac: int,
    nummap: int,
    numepochs: int,
    rng: np.random.Generator,
    verbose: bool,
    subspace_dims: int = 2,
) -> tuple[dict[str, np.ndarray], list[float]]:
    wxf = _fit_projection(x, numfac, rng)
    wyf = _fit_projection(y, numfac, rng)
    wpf = _build_pool_matrix(numfac, subspace_dims)
    whf = np.zeros((nummap, wpf.shape[0]), dtype=np.float64)
    bmap = np.zeros(nummap, dtype=np.float64)
    history: list[float] = []
    for epoch in range(max(1, numepochs)):
        fx = x @ wxf
        fy = y @ wyf
        pooled = _softplus(fx * fy) @ wpf.T
        whf, bmap = _fit_mapping(pooled, nummap, rng)
        gates = _sigmoid(pooled @ whf.T + bmap)
        factors_h = (gates @ whf) @ wpf
        wxf = 0.8 * wxf + 0.2 * _solve_decoder(fy * factors_h, x)
        wyf = 0.8 * wyf + 0.2 * _solve_decoder(fx * factors_h, y)
        error = _reconstruction_error(x, y, wxf, wyf, whf, bmap, wpf=wpf)
        history.append(error)
        if verbose and (epoch == 0 or (epoch + 1) % 10 == 0 or epoch == numepochs - 1):
            print(f"epoch {epoch + 1}: reconstruction_error={error:.6f}")
    return {"wxf": wxf, "wyf": wyf, "whf": whf, "wpf": wpf, "bmap": bmap}, history


def _prepare_features(x: np.ndarray, y: np.ndarray, do_norm: bool) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("Input matrices 'x' and 'y' must be 2D")
    if x.shape[0] != y.shape[0]:
        raise ValueError("Input matrices 'x' and 'y' must have the same number of rows")
    if do_norm:
        eps = 1e-12
        x = x - x.mean(axis=0, keepdims=True)
        y = y - y.mean(axis=0, keepdims=True)
        x = x / (x.std(axis=0, keepdims=True) + x.std() * 0.1 + eps)
        y = y / (y.std(axis=0, keepdims=True) + y.std() * 0.1 + eps)
    return x, y


def _extract_input_pair_matrices(mat: dict, input_file: Path) -> tuple[np.ndarray, np.ndarray]:
    if "x" in mat and "y" in mat:
        return mat["x"], mat["y"]
    if "lfeat" in mat and "rfeat" in mat:
        return mat["lfeat"], mat["rfeat"]
    raise ValueError(
        f"Input .mat file must contain either ('x', 'y') or ('lfeat', 'rfeat') matrices: {input_file}"
    )


@dataclass
class GaeResult:
    algorithm: str
    variant: str
    input_path: str
    output_path: str
    numfac: int
    nummap: int
    numepochs: int
    reconstruction_error: float
    history: list[float]


def run_gae(
    input_path: str,
    output_path: str | None = None,
    variant: str = "standard",
    numfac: int = 600,
    nummap: int = 400,
    learnrate: float = 0.01,
    numepochs: int = 100,
    donorm: bool = True,
    verbose: bool = True,
    random_state: int = 1,
    subspace_dims: int = 2,
) -> GaeResult:
    variant = variant.lower()
    if variant not in {"standard", "multiview"}:
        raise ValueError("variant must be 'standard' or 'multiview'")
    input_file = resolve_user_path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    output_file = (
        workspace_root() / output_path
        if output_path and not Path(output_path).is_absolute()
        else Path(output_path) if output_path
        else input_file.with_name(f"{input_file.stem}_{variant}.mat")
    )
    mat = sio.loadmat(input_file)
    x_raw, y_raw = _extract_input_pair_matrices(mat, input_file)
    x, y = _prepare_features(x_raw, y_raw, do_norm=donorm)
    rng = np.random.default_rng(random_state)
    if variant == "standard":
        payload, history = _standard_gae(
            x=x,
            y=y,
            numfac=numfac,
            nummap=nummap,
            learnrate=learnrate,
            numepochs=numepochs,
            rng=rng,
            verbose=verbose,
        )
    else:
        payload, history = _multiview_gae(
            x=x,
            y=y,
            numfac=numfac,
            nummap=nummap,
            numepochs=numepochs,
            rng=rng,
            verbose=verbose,
            subspace_dims=subspace_dims,
        )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    sio.savemat(output_file, payload, oned_as="column")
    return GaeResult(
        algorithm="gae",
        variant=variant,
        input_path=str(input_file),
        output_path=str(output_file),
        numfac=numfac,
        nummap=nummap,
        numepochs=numepochs,
        reconstruction_error=float(history[-1]),
        history=[float(item) for item in history],
    )
