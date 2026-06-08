from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MplPath
from scipy.interpolate import griddata


ROOT = Path(__file__).resolve().parent
COMSOL_DIR = ROOT / "comsol"
PRED_ROOT = ROOT / "3.1baseline" / "RBF" / "after" / "3.3.3"
K0_DEFAULT = 2.0


def load_table(path: Path, comment_prefix: str) -> np.ndarray:
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith(comment_prefix):
                continue
            rows.append([float(x) for x in s.replace(",", " ").split()])
    if not rows:
        raise ValueError(f"No numeric rows found in {path}")
    return np.asarray(rows, dtype=float)


def parse_k0(pred_path: Path) -> float:
    with pred_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s.startswith("#"):
                break
            if "Wave number k0:" in s:
                return float(s.split("Wave number k0:")[1].strip())
    return K0_DEFAULT


def parse_vertices(pred_path: Path):
    with pred_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s.startswith("#"):
                break
            content = s.lstrip("#").strip()
            if content.startswith("vertices="):
                import ast

                return np.asarray(ast.literal_eval(content.split("vertices=", 1)[1].strip()), dtype=float)
    return None


def grid_field(xy: np.ndarray, values: np.ndarray, xi: np.ndarray, yi: np.ndarray):
    xg, yg = np.meshgrid(xi, yi)
    zg = griddata((xy[:, 0], xy[:, 1]), values, (xg, yg), method="linear")
    if np.any(np.isnan(zg)):
        nearest = griddata((xy[:, 0], xy[:, 1]), values, (xg, yg), method="nearest")
        zg[np.isnan(zg)] = nearest[np.isnan(zg)]
    return xg, yg, zg


def mask_scatterer(pred_path: Path, xg: np.ndarray, yg: np.ndarray):
    verts = parse_vertices(pred_path)
    if verts is None or len(verts) < 3:
        return np.zeros_like(xg, dtype=bool)
    points = np.column_stack([xg.ravel(), yg.ravel()])
    return MplPath(verts).contains_points(points).reshape(xg.shape)


def electric_prediction(pred_path: Path, xi: np.ndarray, yi: np.ndarray):
    arr = load_table(pred_path, "#")
    xy = arr[:, :2]
    ez_scat = arr[:, 2] + 1j * arr[:, 3]
    k0 = parse_k0(pred_path)
    amp = np.abs(ez_scat + np.exp(1j * k0 * xy[:, 0]))
    return grid_field(xy, amp, xi, yi)


def electric_reference(comsol_path: Path, xi: np.ndarray, yi: np.ndarray):
    arr = load_table(comsol_path, "%")
    xy = arr[:, :2]
    ez = arr[:, 2] + 1j * arr[:, 3]
    return grid_field(xy, np.abs(ez), xi, yi)


def draw_case(name: str, comsol_file: Path, prediction_files):
    all_paths = [comsol_file] + [p for _, p in prediction_files]
    for path in all_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    ref_arr = load_table(comsol_file, "%")
    xy_ref = ref_arr[:, :2]
    xi = np.linspace(xy_ref[:, 0].min(), xy_ref[:, 0].max(), 260)
    yi = np.linspace(xy_ref[:, 1].min(), xy_ref[:, 1].max(), 260)

    xg, yg, ref = electric_reference(comsol_file, xi, yi)
    fields = [("COMSOL reference", ref)]
    errors = []

    for label, pred_path in prediction_files:
        _, _, pred = electric_prediction(pred_path, xi, yi)
        interior = mask_scatterer(pred_path, xg, yg)
        pred = pred.copy()
        pred[interior] = np.nan
        err = np.abs(pred - ref)
        fields.append((label, pred))
        errors.append((f"|{label} - reference|", err))

    ref = ref.copy()
    ref[np.isnan(fields[1][1])] = np.nan
    fields[0] = (fields[0][0], ref)

    panels = fields + errors
    ncols = len(panels)
    fig, axes = plt.subplots(1, ncols, figsize=(3.2 * ncols, 3.0), constrained_layout=True)
    if ncols == 1:
        axes = [axes]

    vmin = np.nanmin([np.nanmin(z) for _, z in fields])
    vmax = np.nanmax([np.nanmax(z) for _, z in fields])
    err_max = np.nanmax([np.nanmax(z) for _, z in errors]) if errors else vmax

    for ax, (title, z) in zip(axes, panels):
        is_error = title.startswith("|")
        im = ax.imshow(
            z,
            origin="lower",
            extent=[xi.min(), xi.max(), yi.min(), yi.max()],
            cmap="magma" if is_error else "viridis",
            vmin=0 if is_error else vmin,
            vmax=err_max if is_error else vmax,
        )
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    out = ROOT / f"engineering_{name}_electric_comparison.png"
    fig.savefig(out, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    draw_case(
        "tank",
        COMSOL_DIR / "tank.txt",
        [("BCT", PRED_ROOT / "tank" / "prediction_data.txt")],
    )
    draw_case(
        "airplane",
        COMSOL_DIR / "airplane.txt",
        [
            ("BCT without data", PRED_ROOT / "airplane" / "nodata" / "prediction_data.txt"),
            ("BCT with data", PRED_ROOT / "airplane" / "data" / "prediction_data.txt"),
        ],
    )


if __name__ == "__main__":
    main()
