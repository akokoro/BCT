"""
After-Transformation Error Comparison Figure
==============================================
Layout (6 rows  6 cols):
  Row 0     : Ground Truth (COMSOL reference)
               polygon cases x-mirrored; ellipse cases NOT mirrored
  Rows 1-5  : Absolute Error Metrics |GT - Pred| for each model
              (PINN, GPINN, RoPINN, VSPiNN, RBF)

Colorbars: each column has its own colorbar on its right edge
  GT row   jet colormap, vmin/vmax per case
  Error rows  custom warm colormap, shared 0~err_cap scale
"""

import os
import numpy as np
import ast
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from scipy.interpolate import griddata
from matplotlib.path import Path
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.patches import Ellipse as MplEllipse

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'SimSun'],
    'axes.linewidth': 0.5,
})

#  PATHS 
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
COMSOL_DIR = os.path.join(DATA_DIR, 'comsol')
BASE_DIR   = os.path.join(DATA_DIR, '3.1baseline')

CASES = [
    'polygon_k2_IQ=0.9',
    'polygon_k2_IQ=0.8',
    'polygon_k2_IQ=0.7',
    'polygon_k2_IQ=0.6',
    'shape_L',
    'ellipse_k2_1,0.290',
]
CASE_LABELS = [
    ' regular hexagon',
    ' square',
    ' rectangle',
    ' triangle',
    'L-shape',
    ' ellipse',
]

# COMSOL setup here originally used left-to-right properly, so no need to mirror anymore
MIRROR_X = [False, False, False, False, False, False]

MODELS       = ['pinn', 'gpinn', 'Ropinn', 'VSpinn', 'RBF']
MODEL_LABELS = ['PINN', 'GPINN', 'RoPINN', 'VSPiNN', 'RBF']
MODEL_COLORS = ['#7ec8e3', '#ffd580', '#b5ead7', '#ffb7b2', '#c9b1ff']

POLYGON_VERTICES = {
    'polygon_k2_IQ=0.9': np.array(
        [[1, 0], [0.5, np.sqrt(3) / 2], [-0.5, np.sqrt(3) / 2], [-1, 0], [-0.5, -np.sqrt(3) / 2], [0.5, -np.sqrt(3) / 2]]
    ),
    'polygon_k2_IQ=0.8': np.array(
        [[np.sqrt(2) / 2, np.sqrt(2) / 2], [-np.sqrt(2) / 2, np.sqrt(2) / 2], [-np.sqrt(2) / 2, -np.sqrt(2) / 2], [np.sqrt(2) / 2, -np.sqrt(2) / 2]]
    ),
    'polygon_k2_IQ=0.7': np.array([[1.0, 0.5041], [-1.0, 0.5041], [-1.0, -0.5041], [1.0, -0.5041]]),
    'polygon_k2_IQ=0.6': np.array([[0.0, 1.1547005383], [-1.0, -0.5773502692], [1.0, -0.5773502692]]),
    'shape_L': np.array([[-0.9, -0.5], [0.9, -0.5], [0.9, -0.3], [-0.7, -0.3], [-0.7, 0.5], [-0.9, 0.5]]),
}


def resolve_prediction_path(model, phase, case):
    if model == 'RBF':
        return os.path.join(BASE_DIR, model, phase, case, 'prediction_data.txt')
    return os.path.join(BASE_DIR, model, 'results', phase, case, 'prediction_data.txt')


def parse_geometry_from_prediction(path):
    info = {"type": None, "vertices": None, "ellipse_b": None}
    if path is None or not os.path.exists(path):
        return info
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            s = line.strip()
            if not s.startswith('#'):
                break
            if 'Inner Geometry' in s:
                continue
            if 'Type: Polygon' in s:
                info["type"] = "polygon"
            elif 'Type: Ellipse' in s:
                info["type"] = "ellipse"
            elif 'vertices=' in s:
                try:
                    info["vertices"] = np.asarray(ast.literal_eval(s.split('vertices=')[1].strip()), dtype=float)
                except Exception:
                    pass
            elif 'semiminor=' in s:
                try:
                    val = s.split('semiminor=')[1].split(',')[0].strip()
                    info["ellipse_b"] = float(val)
                except Exception:
                    pass
    return info

#  Colormaps 
CMAP_FIELD = 'jet'
CMAP_ERROR = 'jet'

DARK_BG   = 'white'
SPINE_COL = 'black'
LABEL_COL = 'black'

NC = len(CASES)
NM = len(MODELS)
NR = 1 + NM

#  DATA READERS 

def read_comsol(path, mirror_x=False):
    """COMSOL data -> (x, y, amp_total).
       COMSOL output format assumes col 0:x, col 1:y, col 2:real(Ez), col 3:imag(Ez).
    """
    rows = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            s = line.strip()
            if not s or s[0] == '%':
                continue
            try:
                v = [float(t) for t in s.split()]
                if len(v) >= 4:
                    rows.append(v)
            except ValueError:
                pass
    a = np.asarray(rows)
    x = (-a[:, 0] if mirror_x else a[:, 0])
    y = a[:, 1]
    
    u_c = a[:, 2] + 1j * a[:, 3]
    amp_total = np.abs(u_c)
    return x, y, amp_total


def read_pred(path, k0=2.0):
    """prediction_data.txt  (x, y, |u_total|, |u_scat|).
       Prediction stores the scattered field; total = scatter + e^(j*k0*x).
    """
    rows = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            s = line.strip()
            if not s or s[0] == '#':
                continue
            try:
                v = [float(t) for t in s.split()]
                if len(v) >= 4:
                    rows.append(v)
            except ValueError:
                pass
    a = np.asarray(rows)
    x = a[:, 0]
    y = a[:, 1]
    u_scat = a[:, 2] + 1j * a[:, 3]
    u_inc  = np.exp(1j * k0 * x)
    u_tot  = u_scat + u_inc
    return x, y, np.abs(u_tot), np.abs(u_scat)


#  LOAD DATA 
print("Loading COMSOL ground truth ")
gt_raw = {}
for case, mirror in zip(CASES, MIRROR_X):
    p = os.path.join(COMSOL_DIR, case + '.txt')
    if os.path.exists(p):
        x, y, amp_tot = read_comsol(p, mirror_x=mirror)
        gt_raw[case] = (x, y, amp_tot)
        print(f"  {case}: total=[{amp_tot.min():.3f},{amp_tot.max():.3f}]  mirror={mirror}")
    else:
        print(f"  !! missing GT for {case}")
        gt_raw[case] = None

print("\nLoading model predictions ")
pred_raw = {}
case_geometry = {case: {"type": None, "vertices": POLYGON_VERTICES.get(case), "ellipse_b": None} for case in CASES}
for model in MODELS:
    for case in CASES:
        p = resolve_prediction_path(model, 'after', case)
        if p is not None and os.path.exists(p):
            pred_raw[(model, case)] = read_pred(p)   # (x, y, amp_tot, amp_scat)
            meta = parse_geometry_from_prediction(p)
            if meta["type"] is not None:
                case_geometry[case]["type"] = meta["type"]
            if meta["vertices"] is not None and case_geometry[case]["vertices"] is None:
                case_geometry[case]["vertices"] = meta["vertices"]
            if meta["ellipse_b"] is not None:
                case_geometry[case]["ellipse_b"] = meta["ellipse_b"]
        else:
            pred_raw[(model, case)] = None
            print(f"  !! missing prediction for {model}/{case}")
    print(f"  {model}: done")


#  INTERIOR MASKS 
print("\nBuilding interior masks ")

GRID_N = 300
interior_masks = {}
common_grids   = {}

for case in CASES:
    if gt_raw[case] is None:
        continue
    x_gt, y_gt, _ = gt_raw[case]
    xi = np.linspace(x_gt.min(), x_gt.max(), GRID_N)
    yi = np.linspace(y_gt.min(), y_gt.max(), GRID_N)
    XI, YI = np.meshgrid(xi, yi)
    common_grids[case] = (XI, YI)

    pts = np.column_stack((XI.ravel(), YI.ravel()))
    
    geom = case_geometry.get(case, {})
    verts = geom.get("vertices")
    if verts is not None:
        path = Path(verts)
        mask = path.contains_points(pts).reshape(XI.shape)
    else:
        b = geom.get("ellipse_b")
        if b is None and case.startswith('ellipse_k2_1,'):
            b = float(case.split(',')[-1])
        if b is not None:
            mask = (XI**2 / 1.0**2 + YI**2 / b**2) <= 1.0
        else:
            mask = np.zeros_like(XI, dtype=bool)
        
    interior_masks[case] = mask
    print(f"  {case}: mask applied.")

#  GRIDDED DATA 
print("\nGridding ")

def scatter_to_grid(x, y, z, XI, YI, imask):
    ZI = griddata((x, y), z, (XI, YI), method='linear')
    if np.isnan(ZI).any():
        ZI_nn = griddata((x, y), z, (XI, YI), method='nearest')
        ZI[np.isnan(ZI)] = ZI_nn[np.isnan(ZI)]
    ZI[imask] = np.nan
    return ZI

gt_grid        = {}
pred_grid      = {} 
pred_scat_grid = {}

for case in CASES:
    if gt_raw[case] is None:
        gt_grid[case] = None
        continue
    XI, YI = common_grids[case]
    imask = interior_masks[case]
    x, y, amp = gt_raw[case]
    gt_grid[case] = scatter_to_grid(x, y, amp, XI, YI, imask)

for model in MODELS:
    for case in CASES:
        raw = pred_raw[(model, case)]
        if raw is None or case not in common_grids:
            pred_grid[(model, case)]      = None
            pred_scat_grid[(model, case)] = None
            continue
        XI, YI = common_grids[case]
        imask = interior_masks[case]
        x, y, amp_tot, amp_scat = raw
        pred_grid[(model, case)]      = scatter_to_grid(x, y, amp_tot,  XI, YI, imask)
        pred_scat_grid[(model, case)] = scatter_to_grid(x, y, amp_scat, XI, YI, imask)

print("  done")

#  COLOR LIMITS 
field_vlim = {}
for case in CASES:
    g = gt_grid.get(case)
    if g is not None:
        g_masked = g.copy()
        g_masked[interior_masks[case]] = np.nan
        field_vlim[case] = (float(np.nanpercentile(g_masked, 2)), float(np.nanpercentile(g_masked, 98)))
    else:
        field_vlim[case] = (0.0, 1.0)


#  FIGURE LAYOUT 
ROW_H = [1.0] * NR

FIG_W = 14.0

LABEL_LEFT   = 0.075
USABLE_RIGHT = 0.995
TOP = 0.925
BOT = 0.022

GAP_FRAC  = 0.003
CB_W      = 0.009
INTER_COL = 0.016
HSPACE  = 0.014

usable   = USABLE_RIGHT - LABEL_LEFT
col_unit = usable / NC
sp_w     = col_unit - GAP_FRAC - CB_W - INTER_COL

sp_w_inch          = sp_w * FIG_W
effective_row_frac = (TOP - BOT) / NR - HSPACE
FIG_H = sp_w_inch / effective_row_frac
FIG_H = max(9.0, min(FIG_H, 18.0))

total_h = sum(ROW_H)
hr = [h / total_h for h in ROW_H]

fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor='white')

col_sp_left = []
col_cb_left = []

for ci in range(NC):
    sp_l = LABEL_LEFT + ci * col_unit
    cb_l = sp_l + sp_w + GAP_FRAC
    col_sp_left.append(sp_l)
    col_cb_left.append(cb_l)

row_tops = []
row_bots = []
cumulative = 0.0
for frac in hr:
    cumulative += frac
    row_tops.append(TOP - (cumulative - frac) * (TOP - BOT))
    row_bots.append(TOP - cumulative * (TOP - BOT))

adjusted_tops = []
adjusted_bots = []
for ri in range(NR):
    t = row_tops[ri] - (HSPACE / 2 if ri > 0 else 0)
    b = row_bots[ri] + (HSPACE / 2 if ri < NR - 1 else 0)
    adjusted_tops.append(t)
    adjusted_bots.append(b)

axes    = {}
for ri in range(NR):
    for ci in range(NC):
        rect = [col_sp_left[ci], adjusted_bots[ri],
                sp_w,            adjusted_tops[ri] - adjusted_bots[ri]]
        ax = fig.add_axes(rect)
        axes[(ri, ci)] = ax

#  DRAW HELPER 
def draw_cell(ax, ZI, vmin, vmax, cmap, case=None):
    ax.set_facecolor('white')
    if ZI is None:
        ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                color='#ff4444', transform=ax.transAxes, fontsize=7)
    else:
        XI, YI = common_grids[case]
        ZI_clipped = np.clip(ZI, vmin, vmax)
        levels = np.linspace(vmin, vmax, 50)
        ax.contourf(XI, YI, ZI_clipped, levels=levels, cmap=cmap)
        
        # Draw interior masks cleanly
        geom = case_geometry.get(case, {})
        vertices = geom.get("vertices")
        if vertices is not None:
            patch = MplPolygon(vertices, facecolor='white', edgecolor='black', linewidth=1.5)
            ax.add_patch(patch)
        else:
            semiminor = geom.get("ellipse_b")
            if semiminor is None and case and 'ellipse' in case:
                semiminor = float(case.split(',')[-1])
            if semiminor is not None:
                patch = MplEllipse((0, 0), 2.0, 2 * semiminor, facecolor='white', edgecolor='black', linewidth=1.5)
                ax.add_patch(patch)

    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor('black')
        sp.set_linewidth(0.6)

#  POPULATE CELLS & COLORBARS 
def add_cell_colorbar(ax_cb, sm, is_error=False):
    cb = fig.colorbar(sm, cax=ax_cb)
    ax_cb.set_visible(False)

for j, (case, clbl) in enumerate(zip(CASES, CASE_LABELS)):
    ax = axes[(0, j)]
    cb_l = col_cb_left[j]
    t, b = adjusted_tops[0], adjusted_bots[0]
    ax_cb = fig.add_axes([cb_l, b, CB_W, t - b])
    
    if gt_grid.get(case) is not None:
        vm, vM = field_vlim[case]
        draw_cell(ax, gt_grid[case], vm, vM, CMAP_FIELD, case=case)
        sm = ScalarMappable(norm=Normalize(vm, vM), cmap=CMAP_FIELD)
        sm.set_array([])
        add_cell_colorbar(ax_cb, sm, is_error=False)
    else:
        draw_cell(ax, None, 0, 1, CMAP_FIELD, case=case)
        ax_cb.set_visible(False)
        
    ax.set_title(clbl, color='black', fontsize=14, pad=3,
                 fontweight='bold', linespacing=1.3)


for mi, model in enumerate(MODELS):
    row = mi + 1
    t, b = adjusted_tops[row], adjusted_bots[row]
    for j, case in enumerate(CASES):
        p = pred_grid.get((model, case))
        g = gt_grid.get(case)
        ax = axes[(row, j)]
        cb_l = col_cb_left[j]
        ax_cb = fig.add_axes([cb_l, b, CB_W, t - b])
        
        if p is not None and g is not None:
            p_masked = p.copy()
            p_masked[interior_masks[case]] = np.nan
            
            if np.sum(~np.isnan(p_masked)) > 0:
                vm, vM = np.nanpercentile(p_masked, 2), np.nanpercentile(p_masked, 98)
            else:
                vm, vM = 0.0, 1.0
            if np.isnan(vm) or np.isnan(vM) or vm == vM:
                vm, vM = 0.0, 1.0
            
            draw_cell(ax, p_masked, vm, vM, CMAP_FIELD, case=case)
            sm = ScalarMappable(norm=Normalize(vm, vM), cmap=CMAP_FIELD)
            sm.set_array([])
            add_cell_colorbar(ax_cb, sm, is_error=False)
            
            p_tot  = pred_grid[(model, case)]
            g_tot  = gt_grid[case]
            valid = (~np.isnan(p_tot) & ~np.isnan(g_tot)
                     & ~interior_masks[case])
            if np.sum(valid) > 0:
                num = np.sum(np.abs(p_tot[valid] - g_tot[valid]))
                den = np.sum(np.abs(g_tot[valid]))
                rel_err = (num / den) * 100.0 if den > 0 else 0.0
                ax.text(0.97, 0.96, f"{rel_err:.2f}%",
                        transform=ax.transAxes, ha='right', va='top',
                        color='black', fontsize=13, fontweight='bold',
                        bbox=dict(facecolor='white', alpha=0.90, edgecolor='black',
                                  linestyle='--', linewidth=1.0, boxstyle='square,pad=0.20'))
        else:
            draw_cell(ax, None, 0, 1, CMAP_FIELD, case=case)
            ax_cb.set_visible(False)


#  ROW LABELS 
def row_yctr(ri):
    return (adjusted_tops[ri] + adjusted_bots[ri]) / 2

x_name = LABEL_LEFT - 0.010
x_err  = LABEL_LEFT - 0.038

fig.text(x_name, row_yctr(0), 'Numerical solution',
         color='black', fontsize=14, va='center', ha='center',
         fontweight='bold', rotation=90)

for mi, (mlbl, _) in enumerate(zip(MODEL_LABELS, MODEL_COLORS)):
    row = mi + 1
    fig.text(x_name, row_yctr(row), mlbl,
             color='black', fontsize=14, va='center', ha='center',
             fontweight='bold', rotation=90)

#  SAVE 
out_pdf = os.path.join(DATA_DIR, 'after_comparison.pdf')
out_png = os.path.join(DATA_DIR, 'after_comparison.png')
plt.savefig(out_pdf, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(out_png, dpi=200, bbox_inches='tight', facecolor='white')
print(f"\nSaved:\n  {out_pdf}\n  {out_png}")
