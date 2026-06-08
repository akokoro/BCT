from shape import Shape
import numpy as np
import torch
from scipy.special import jv, hankel1
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Ellipse as MplEllipse, Polygon as MplPolygon
import rbf_net
import time
import os

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:", device)

# Training control
TRAIN_MODEL = False  # Set to False to load existing model and visualize only

# General parameters
weights = 1
iterations = 6000
learning_rate = 1e-3

# Data-driven settings
DATA_PERCENTAGE = 25.0
data_weight = 6400.0

# Problem parameters
k0 = 2
wave_len = 2 * np.pi / k0
length = 2 * np.pi
L = np.pi / 2
n_wave = 30
h_elem = wave_len / n_wave
nx = int(length / h_elem)

# RBF parameters
n_in = 2
n_out = 2  # Only E field (real and imaginary parts)
b = 5
n_neu_x = nx + 6
n_neu_y = nx + 6
c_x = [-length / 2 - length / 20, length / 2 + length / 20]
c_y = [-length / 2 - length / 20, length / 2 + length / 20]



def get_geometry_info(geometry, label, _depth=0):
    """Return a text description of a geometry for prediction_data.txt."""
    if _depth > 8:
        return f"# {label}: <nesting too deep>\n"
    info = f"# {label}:\n"
    if isinstance(geometry, Shape.GeometryDifference):
        info += "#   Type: GeometryDifference (domain / scatterer = A minus B)\n"
        info += get_geometry_info(geometry.geom_a, f"{label}  A", _depth + 1)
        info += get_geometry_info(geometry.geom_b, f"{label}  B", _depth + 1)
        return info
    if isinstance(geometry, Shape.RectangleGeometry):
        info += "#   Type: Rectangle\n"
        info += f"#   xmin={geometry.xmin}, xmax={geometry.xmax}\n"
        info += f"#   ymin={geometry.ymin}, ymax={geometry.ymax}\n"
    elif isinstance(geometry, Shape.Disk):
        info += "#   Type: Disk (Circle)\n"
        info += f"#   center=({geometry.center[0]}, {geometry.center[1]})\n"
        info += f"#   radius={geometry.radius}\n"
    elif isinstance(geometry, Shape.Ellipse):
        info += "#   Type: Ellipse\n"
        info += f"#   center=({geometry.center[0]}, {geometry.center[1]})\n"
        info += f"#   semimajor={geometry.semimajor}, semiminor={geometry.semiminor}\n"
        info += f"#   rotation_angle={geometry.angle} degrees\n"
    elif isinstance(geometry, Shape.Polygon):
        info += "#   Type: Polygon\n"
        info += f"#   num_vertices={geometry.nvertices}\n"
        info += f"#   vertices={geometry.vertices.tolist()}\n"
    elif isinstance(geometry, Shape.StarShaped):
        info += "#   Type: Star-shaped\n"
        info += f"#   center=({geometry.center[0]}, {geometry.center[1]})\n"
        info += f"#   num_points={geometry.num_points}\n"
        info += f"#   inner_radius={geometry.inner_radius}, outer_radius={geometry.outer_radius}\n"
    elif isinstance(geometry, Shape.SuperEllipse):
        info += "#   Type: SuperEllipse\n"
        info += f"#   center={geometry.center.tolist()}, a={geometry.a}, b={geometry.b}, n={geometry.n}, angle={geometry.angle}\n"
    elif isinstance(geometry, Shape.RadialFourier):
        info += "#   Type: RadialFourier\n"
        info += f"#   center={geometry.center.tolist()}, r0={geometry.r0}, a={geometry.a}, m={geometry.m}, angle={geometry.angle}\n"
    else:
        info += f"#   Type: {type(geometry).__name__}\n"
    return info


def _stroke_geometry_boundary(ax, geometry):
    """"""
    if isinstance(geometry, Shape.RectangleGeometry):
        xs = [
            geometry.xmin,
            geometry.xmax,
            geometry.xmax,
            geometry.xmin,
            geometry.xmin,
        ]
        ys = [
            geometry.ymin,
            geometry.ymin,
            geometry.ymax,
            geometry.ymax,
            geometry.ymin,
        ]
        ax.plot(xs, ys, "k-", linewidth=1.5, zorder=20)
    elif isinstance(geometry, Shape.Disk):
        t = np.linspace(0.0, 2.0 * np.pi, 200, endpoint=False)
        ax.plot(
            geometry.center[0] + geometry.radius * np.cos(t),
            geometry.center[1] + geometry.radius * np.sin(t),
            "k-",
            linewidth=1.5,
            zorder=20,
        )
    elif isinstance(geometry, Shape.Ellipse):
        th = np.linspace(0.0, 2.0 * np.pi, 200, endpoint=False)
        pl = np.stack(
            [geometry.semimajor * np.cos(th), geometry.semiminor * np.sin(th)], axis=1
        )
        xy = pl @ geometry.rot_matrix.T + geometry.center
        ax.plot(xy[:, 0], xy[:, 1], "k-", linewidth=1.5, zorder=20)
    elif isinstance(geometry, (Shape.Polygon, Shape.StarShaped)):
        v = np.vstack([geometry.vertices, geometry.vertices[0]])
        ax.plot(v[:, 0], v[:, 1], "k-", linewidth=1.5, zorder=20)
    elif isinstance(geometry, (Shape.SuperEllipse, Shape.RadialFourier)):
        bd = getattr(geometry, "boundary_pts", None)
        if bd is not None and len(bd) > 2:
            v = np.vstack([bd, bd[0]])
            ax.plot(v[:, 0], v[:, 1], "k-", linewidth=1.5, zorder=20)
    else:
        try:
            b = geometry.random_boundary_points(500)
            ax.plot(b[:, 0], b[:, 1], "k-", linewidth=1.5, zorder=20)
        except Exception:
            pass


# ==================== Data Loading Functions ====================
def load_numerical_data(filepath, k0):
    """
    
    Args:
        filepath: 
        k0: 
    Returns:
        coords:  [N, 2]
        values:  [N, 2] (, )
    """
    data = np.loadtxt(filepath, skiprows=9)
    coords = data[:, :2]
    values = data[:, 2:4]
    
    incident_field = np.exp(1j * k0 * coords[:, 0:1])
    incident_real = np.real(incident_field).flatten()
    incident_imag = np.imag(incident_field).flatten()
    
    values[:, 0] = values[:, 0] - incident_real
    values[:, 1] = values[:, 1] - incident_imag
    
    return coords, values
# ==================== Exact Solution ====================
def sound_hard_circle_deepxde(k0, a, points):
    """Analytical solution for sound-hard circle scattering"""
    fem_xx = points[:, 0:1]
    fem_xy = points[:, 1:2]
    r = np.sqrt(fem_xx * fem_xx + fem_xy * fem_xy)
    theta = np.arctan2(fem_xy, fem_xx)
    npts = np.size(fem_xx, 0)
    n_terms = int(30 + (k0 * a) ** 1.01)

    u_sc_E = np.zeros((npts), dtype=np.complex128)
    u_sc_H = np.zeros((npts), dtype=np.complex128)
    for n in range(-n_terms, n_terms):
        bessel_deriv = jv(n - 1, k0 * a) - n / (k0 * a) * jv(n, k0 * a)
        hankel_deriv = n / (k0 * a) * hankel1(n, k0 * a) - hankel1(n + 1, k0 * a)
        u_sc_H += (
            -((1j) ** (n))
            * (bessel_deriv / hankel_deriv)
            * hankel1(n, k0 * r)
            * np.exp(1j * n * theta)
        ).ravel()

        u_sc_E += (
                -((1j) ** (n))
                * (jv(n, k0 * a) / hankel1(n, k0 * a))
                * hankel1(n, k0 * r)
                * np.exp(1j * n * theta)
        ).ravel()
    return u_sc_E,u_sc_H

def sol_H(x):
    """Get analytical solution for H field (real and imaginary parts)"""
    result = sound_hard_circle_deepxde(k0, L/2, x)[1].reshape((x.shape[0], 1))
    real = np.real(result)
    imag = np.imag(result)
    return np.hstack((real, imag))

def sol_E(x):
    """Get analytical solution for E field (real and imaginary parts)"""
    # For TM mode, E_z = u_sc where u_sc is the scattered field
    # E field has Dirichlet BC: E_total = 0 on conductor => E_sc = -E_inc
    result = sound_hard_circle_deepxde(k0, L/2, x)[0].reshape((x.shape[0], 1))
    real = np.real(result)
    imag = np.imag(result)
    return np.hstack((real, imag))

def sol(x):
    """Get analytical solution for both E and H fields (4 components)"""
    # Returns: [E_real, E_imag, H_real, H_imag]
    E_sol = sol_E(x)
    H_sol = sol_H(x)
    return np.hstack((E_sol, H_sol))

# ==================== PDE and Boundary Condition Functions ====================
def pde_residual(x, y, k0):
    y0 = y[:, 0:1]  # real part
    y1 = y[:, 1:2]  # imaginary part

    grad_y0 = torch.autograd.grad(y0.sum(), x, create_graph=True)[0]
    grad_y1 = torch.autograd.grad(y1.sum(), x, create_graph=True)[0]

    laplacian_y0 = 0
    laplacian_y1 = 0
    for i in range(2):
        grad2_y0 = torch.autograd.grad(grad_y0[:, i].sum(), x, create_graph=True)[0][:, i:i + 1]
        grad2_y1 = torch.autograd.grad(grad_y1[:, i].sum(), x, create_graph=True)[0][:, i:i + 1]
        laplacian_y0 += grad2_y0
        laplacian_y1 += grad2_y1

    residual_0 = laplacian_y0 + k0 ** 2 * y0
    residual_1 = laplacian_y1 + k0 ** 2 * y1

    return [residual_0, residual_1]

def compute_normal_derivative(y, x, normal, component):
    """
    Compute normal derivative: dy/dn = y  n
    
    Args:
        y: output tensor
        x: input tensor
        normal: normal vectors [batch, 2]
        component: which output component
    """
    grad_outputs = torch.ones_like(y[:, component:component+1])
    dy_dx = torch.autograd.grad(
        outputs=y[:, component:component+1],
        inputs=x,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True
    )[0]
    
    # Dot product with normal
    dy_dn = dy_dx[:, 0:1] * normal[:, 0:1] + dy_dx[:, 1:2] * normal[:, 1:2]
    return dy_dn

def compute_H_from_E(E_real, E_imag, x, k0):
    """
    Compute H field from E field using Maxwell's equations.
    For 2D TM mode: The relationship is H = (1/(j)) * E
    For sound-hard scattering with Neumann BC: dH/dn = jk0*E_incn
    
    Args:
        E_real: real part of E field [batch, 1]
        E_imag: imaginary part of E field [batch, 1]
        x: input coordinates [batch, 2]
        k0: wave number
    
    Returns:
        H_real, H_imag: real and imaginary parts of H field
    """
    # Compute gradients of E
    grad_E_real = torch.autograd.grad(E_real.sum(), x, create_graph=True, retain_graph=True)[0]
    grad_E_imag = torch.autograd.grad(E_imag.sum(), x, create_graph=True, retain_graph=True)[0]
    
    # For the sound-hard scattering problem:
    # H = (1/(jk0)) * E = (1/(jk0)) * (E/x, E/y)
    # H_real  (1/k0) * E_imag/x
    # H_imag  -(1/k0) * E_real/x
    
    dE_real_dx = grad_E_real[:, 0:1]
    dE_imag_dx = grad_E_imag[:, 0:1]
    
    H_real = (1.0 / k0) * dE_imag_dx
    H_imag = -(1.0 / k0) * dE_real_dx
    
    return H_real, H_imag

# Boundary condition functions for E field only
def func0_outer_robin(y):
    """Real part of Robin BC on outer boundary: dy/dn = -k0 * y_imag"""
    return -k0 * y[:, 1:2]

def func1_outer_robin(y):
    """Imaginary part of Robin BC on outer boundary: dy/dn = k0 * y_real"""
    return k0 * y[:, 0:1]

# ==================== Training Loop ====================
def train_model():
    """Main training function"""
    
    # Setup geometry
    # -------------------------------------------------------------------------
    #
    #
    #
    # inner = Shape.polygon_T_shape(stem_half_width=0.12, stem_height=0.90,
    # inner = Shape.polygon_arrow(body_half_width=0.12, body_length=0.70,
    inner = Shape.polygon_airplane_profile(scale=1.0)

    #
    #
    # -------------------------------------------------------------------------
    # vertices = [
    #     [0.000000, -0.618034],
    #     [-0.138757, -0.190983],
    #     [-0.587785, -0.190983],
    #     [-0.224514, 0.072949],
    #     [-0.363271, 0.500000],
    #     [0.000000, 0.236068],
    #     [0.363271, 0.500000],
    #     [0.224514, 0.072949],
    #     [0.587785, -0.190983],
    #     [0.138757, -0.190983]
    # ] # 0.8450, 0.5279
    outer = Shape.RectangleGeometry(-length/2, length/2, -length/2, length/2)

    # vertices = [[1, 0], [1 / 2, np.sqrt(3) / 2], [-1 / 2, np.sqrt(3) / 2], [-1, 0], [-1 / 2, -np.sqrt(3) / 2],
    #             [1 / 2, -np.sqrt(3) / 2]]
    # inner = Shape.Polygon(vertices)
    # inner = Shape.Disk([0, 0], radius=L/2)
    # inner = Shape.RectangleGeometry(-L/2, L/2, -L/2, L/2)

    # inner = Shape.Ellipse([0, 0], semimajor=1.0, semiminor=0.585) # IQ = 0.9,  0.0739, 0.0560
    # inner = Shape.Ellipse([0, 0], semimajor=1.0, semiminor=0.453) # IQ = 0.8, 0.1502, 0.0951
    # inner = Shape.Ellipse([0, 0], semimajor=1.0, semiminor=0.362) # IQ = 0.7, 0.2292, 0.1333
    # inner = Shape.Ellipse([0, 0], semimajor=1.0, semiminor=0.290) # IQ = 0.6, 0.3125, 0.1712
    # inner = Shape.Ellipse([0, 0], semimajor=1.0, semiminor=0.229) # IQ = 0.5, 0.4016, 0.2084
    # inner = Shape.Ellipse([0, 0], semimajor=1.0, semiminor=0.176) # IQ = 0.4, 0.4966, 0.2573
    # inner = Shape.Ellipse([0, 0], semimajor=1.0, semiminor=0.128) # IQ = 0.3,  0.6002, 0.3135

    # vertices = [[1, 0], [0.5, np.sqrt(3) / 2], [-0.5, np.sqrt(3) / 2], [-1, 0], [-0.5, -np.sqrt(3) / 2],
    #             [0.5, -np.sqrt(3) / 2]]  #  0.4447, 0.0207 

    # vertices = [[np.sqrt(2) / 2, np.sqrt(2) / 2], [-np.sqrt(2) / 2, np.sqrt(2) / 2], [-np.sqrt(2) / 2, -np.sqrt(2) / 2],
    #             [np.sqrt(2) / 2, -np.sqrt(2) / 2]]  #  0.5669, 0.0408 

    # inner = Shape.RectangleGeometry(-1, 1, -0.5041, 0.5041)  #  0.5665, 0.1072 

    # vertices = [[0, 2 * np.sqrt(3) / 3], [-1, -np.sqrt(3) / 3], [1, -np.sqrt(3) / 3]]  #  0.6826, 0.0897 
    # inner = Shape.Polygon(vertices) 

    # inner = Shape.RectangleGeometry(-1, 1, -0.2479, 0.2479)  # IQ = 0.5, 0.5666, 0.2229
    #
    # inner = Shape.RectangleGeometry(-1, 1, -0.1761, 0.1761)  # IQ = 0.4, 0.5664, 0.2735
    #
    # inner = Shape.RectangleGeometry(-1, 1, -0.1197, 0.1197)  # IQ = 0.3, 0.5663, 0.3415
    plot_computational_domain_preview(outer, inner)

    geom = Shape.GeometryDifference(outer, inner)
    # Sample training points
    print("Sampling training points...")
    
    from shape import calculate_complexity
    C_local, H_global = calculate_complexity(inner, outer)
    print("=" * 45)
    print(f"C_local, H_global:  {C_local:.4f}, {H_global:.4f}")
    print("=" * 45)

    num_domain = nx ** 2
    num_boundary = 15 * nx
    num_test = nx ** 2
    
    # Domain points (PDE residual)
    domain_points = geom.random_points(num_domain, sampler='Hammersley')
    
    # Boundary points
    boundary_points = geom.random_boundary_points(num_boundary)
    
    # Load numerical simulation data
    data_file = os.path.join(os.path.dirname(__file__), '3.1baseline/comsol', 'airplane.txt')
    numerical_coords, numerical_values = None, None
    data_points_t = None
    data_values_t = None
    num_data_points = 0
    
    if os.path.exists(data_file):
        print(f"Loading numerical data from {data_file}...")
        numerical_coords, numerical_values = load_numerical_data(data_file, k0)

        # Filter points that are inside the computational domain (not inside scatterer)
        inner_geom = geom.geom_b
        mask = np.logical_not(inner_geom.inside(numerical_coords))
        numerical_coords = numerical_coords[mask]
        numerical_values = numerical_values[mask]

        # Calculate required number of data points based on DATA_PERCENTAGE
        # Total collocation points = domain + boundary
        num_collocation = num_domain + num_boundary
        
        if DATA_PERCENTAGE <= 0.0:
            # No data points, pure physics-driven
            n_select = 0
            print(f"DATA_PERCENTAGE = {DATA_PERCENTAGE:.1f}%, using NO data points (pure physics-driven)")
        elif DATA_PERCENTAGE >= 99.0:
            # Use all available data to avoid division by zero
            n_select = len(numerical_coords)
            print(f"DATA_PERCENTAGE = {DATA_PERCENTAGE:.1f}% (>=99%), using all available data points")
        else:
            # Calculate data points: x / (x + num_collocation) = DATA_PERCENTAGE / 100
            # Solve for x: x = num_collocation * DATA_PERCENTAGE / (100 - DATA_PERCENTAGE)
            n_select = int(num_collocation * DATA_PERCENTAGE / (100.0 - DATA_PERCENTAGE))
        
        # Ensure we don't exceed available data
        n_total = len(numerical_coords)
        n_select = min(n_select, n_total)
        
        # Randomly select data points
        if n_select > 0:
            indices = np.random.choice(n_total, n_select, replace=False)
            numerical_coords = numerical_coords[indices]
            numerical_values = numerical_values[indices]
            num_data_points = len(numerical_coords)
            
            # Calculate actual percentage
            actual_percentage = 100.0 * num_data_points / (num_data_points + num_collocation)
            
            # Print detailed statistics
            print(f"\n{'='*60}")
            print("Data Sampling Summary:")
            print(f"{'='*60}")
            print(f"Target DATA_PERCENTAGE:   {DATA_PERCENTAGE:.1f}%")
            print(f"Available data points:    {n_total}")
            print(f"Selected data points:     {num_data_points}")
            print(f"Collocation points:       {num_collocation} (domain: {num_domain}, boundary: {num_boundary})")
            print(f"Total training points:    {num_data_points + num_collocation}")
            print(f"Actual data percentage:   {actual_percentage:.2f}%")
            print(f"{'='*60}\n")
        else:
            numerical_coords = np.array([]).reshape(0, 2)
            numerical_values = np.array([]).reshape(0, 2)
            num_data_points = 0
            print(f"\n{'='*60}")
            print("Training Configuration: Pure Physics-Driven (No Data)")
            print(f"{'='*60}")
            print(f"Collocation points:       {num_collocation} (domain: {num_domain}, boundary: {num_boundary})")
            print(f"Data points:              0")
            print(f"{'='*60}\n")
        
        # Convert data points to tensors (if any)
        if num_data_points > 0:
            data_points_t = torch.tensor(numerical_coords, dtype=torch.float32, device=device, requires_grad=True)
            data_values_t = torch.tensor(numerical_values, dtype=torch.float32, device=device)
    else:
        print(f"Warning: Data file {data_file} not found. Training without numerical data.")
        num_data_points = 0
        data_points_t = None
        data_values_t = None

    # Separate inner and outer boundary points
    inner_mask = inner.on_boundary(boundary_points)
    outer_mask = outer.on_boundary(boundary_points)
    
    bc_inner_points = boundary_points[inner_mask]

    bc_outer_points = boundary_points[outer_mask]
    
    # Compute normals for boundary points
    bc_inner_normals = -inner.boundary_normal(bc_inner_points)  # Inward normal
    bc_outer_normals = geom.boundary_normal(bc_outer_points)
    
    # Convert to tensors
    domain_points_t = torch.tensor(domain_points, dtype=torch.float32, device=device, requires_grad=True)
    bc_inner_points_t = torch.tensor(bc_inner_points, dtype=torch.float32, device=device, requires_grad=True)
    bc_outer_points_t = torch.tensor(bc_outer_points, dtype=torch.float32, device=device, requires_grad=True)
    bc_outer_normals_t = torch.tensor(bc_outer_normals, dtype=torch.float32, device=device)

    # Test points (only for circular geometry)
    is_circle = isinstance(inner, Shape.Disk)
    if is_circle:
        test_points = geom.random_points(num_test, sampler='Hammersley')
        test_points_t = torch.tensor(test_points, dtype=torch.float32, device=device)
        test_solution = sol(test_points)
        test_solution_t = torch.tensor(test_solution, dtype=torch.float32, device=device)
    else:
        test_points_t = None
        test_solution_t = None
    
    print(f"Domain points: {len(domain_points)}")
    print(f"Inner boundary points: {len(bc_inner_points)}")
    print(f"Outer boundary points: {len(bc_outer_points)}")
    if is_circle:
        print(f"Test points: {len(test_points)} (analytical solution available)")
    
    # Initialize network
    net = rbf_net.PINN_module(n_in, n_out, n_neu_x, n_neu_y, b, c_x, c_y, device).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    
    # Training history
    loss_history = {
        'total': [],
        'pde_0': [],
        'pde_1': [],
        'bc_inner_0': [],
        'bc_inner_1': [],
        'bc_outer_0': [],
        'bc_outer_1': [],
        'data_loss': [],
        'test_error_E':[],
        'test_error_H': []
    }
    
    print("\nStarting training...")
    start_time = time.time()
    
    for iteration in range(iterations):
        net.train()
        optimizer.zero_grad()
        
        # ========== PDE Loss (E field only) ==========
        y_domain = net(domain_points_t)
        # E field only (channels 0,1: real and imaginary parts)
        pde_E = pde_residual(domain_points_t, y_domain, k0)
        loss_pde0 = torch.mean(pde_E[0] ** 2)
        loss_pde1 = torch.mean(pde_E[1] ** 2)
        
        # ========== Inner Boundary Loss (Dirichlet BC for E field only) ==========
        y_inner = net(bc_inner_points_t)

        incident_field = np.exp(1j * k0 * bc_inner_points[:, 0:1])
        incident_field_real = np.real(incident_field)
        incident_field_imag = np.imag(incident_field)

        incident_field_real_t = torch.tensor(incident_field_real, dtype=torch.float32, device=device)
        incident_field_imag_t = torch.tensor(incident_field_imag, dtype=torch.float32, device=device)

        loss_bc_inner0 = torch.mean((y_inner[:, 0:1] + incident_field_real_t) ** 2)
        loss_bc_inner1 = torch.mean((y_inner[:, 1:2] + incident_field_imag_t) ** 2)
        
        # ========== Outer Boundary Loss (Robin BC for E field only) ==========
        y_outer = net(bc_outer_points_t)
        
        # Compute normal derivatives
        dy_dn_outer_0 = compute_normal_derivative(y_outer, bc_outer_points_t, bc_outer_normals_t, component=0)
        dy_dn_outer_1 = compute_normal_derivative(y_outer, bc_outer_points_t, bc_outer_normals_t, component=1)
        
        # Robin BC for E field: dy/dn = func(x, y)
        # dy_real/dn = -k0 * y_imag
        # dy_imag/dn = k0 * y_real
        robin_E_real = -k0 * y_outer[:, 1:2]
        robin_E_imag =  k0 * y_outer[:, 0:1]
        loss_bc_outer0 = torch.mean((dy_dn_outer_0 - robin_E_real) ** 2)
        loss_bc_outer1 = torch.mean((dy_dn_outer_1 - robin_E_imag) ** 2)
        
        # ========== Data Loss ==========
        loss_data = torch.tensor(0.0, device=device)
        if data_points_t is not None and num_data_points > 0:
            y_data = net(data_points_t)
            loss_data = torch.mean((y_data[:, 0:2] - data_values_t) ** 2)
        
        # ========== Total Loss ==========
        loss_total = (loss_pde0 + loss_pde1  +
                      weights * loss_bc_inner0 + weights * loss_bc_inner1 +
                     weights * loss_bc_outer0 + weights * loss_bc_outer1 +
                      data_weight * loss_data)
        
        # Backward pass
        loss_total.backward()
        optimizer.step()
        
        # ========== Logging ==========
        loss_history['total'].append(loss_total.item())
        loss_history['pde_0'].append(loss_pde0.item())
        loss_history['pde_1'].append(loss_pde1.item())
        loss_history['bc_inner_0'].append(loss_bc_inner0.item())
        loss_history['bc_inner_1'].append(loss_bc_inner1.item())
        loss_history['bc_outer_0'].append(loss_bc_outer0.item())
        loss_history['bc_outer_1'].append(loss_bc_outer1.item())
        loss_history['data_loss'].append(loss_data.item())
        
        # Compute test error
        if iteration % 100 == 0:
            elapsed = time.time() - start_time
            data_loss_str = f"Data: {loss_data.item():.3e}" if num_data_points > 0 else "Data: N/A"
            
            if is_circle:
                net.eval()
                test_points_req_grad = torch.tensor(test_points, dtype=torch.float32, device=device, requires_grad=True)
                y_test_E_grad = net(test_points_req_grad)
                H_test_real, H_test_imag = compute_H_from_E(
                    y_test_E_grad[:, 0:1], y_test_E_grad[:, 1:2], 
                    test_points_req_grad, k0
                )
                
                with torch.no_grad():
                    # Predict E field
                    y_test_E = net(test_points_t)
                    # Combine E and H for error calculation
                    y_test = torch.cat([y_test_E, H_test_real.detach(), H_test_imag.detach()], dim=1)
                    
                    l2_error_E = torch.sqrt(torch.mean((y_test[:,0:2] - test_solution_t[:,0:2]) ** 2) /
                                         torch.mean(test_solution_t[:,0:2] ** 2))
                    l2_error_H = torch.sqrt(torch.mean((y_test[:,2:] - test_solution_t[:,2:]) ** 2) /
                                          torch.mean(test_solution_t[:,2:] ** 2))

                    loss_history['test_error_E'].append(l2_error_E.item())
                    loss_history['test_error_H'].append(l2_error_H.item())
                
                print(f"Iter {iteration:5d} | Total Loss: {loss_total.item():.6e} | "
                      f"PDE0: {loss_pde0.item():.3e}, {loss_pde1.item():.3e} | "
                      f"BC_in: {loss_bc_inner0.item():.3e}, {loss_bc_inner1.item():.3e} | "
                      f"BC_out: {loss_bc_outer0.item():.3e}, {loss_bc_outer1.item():.3e} | "
                      f"{data_loss_str} | L2 error: {l2_error_E.item() :.6e},{l2_error_H.item():.6e} | Time: {elapsed:.1f}s")
            else:
                print(f"Iter {iteration:5d} | Total Loss: {loss_total.item():.6e} | "
                      f"PDE: {loss_pde0.item():.3e}, {loss_pde1.item():.3e} | "
                      f"BC_in: {loss_bc_inner0.item():.3e}, {loss_bc_inner1.item():.3e} | "
                      f"BC_out: {loss_bc_outer0.item():.3e}, {loss_bc_outer1.item():.3e} | "
                      f"{data_loss_str} | Time: {elapsed:.1f}s")
    
    # Final evaluation
    print(f"\nTraining completed!")
    if is_circle:
        net.eval()
        test_points_req_grad = torch.tensor(test_points, dtype=torch.float32, device=device, requires_grad=True)
        y_test_E_grad = net(test_points_req_grad)
        H_test_real, H_test_imag = compute_H_from_E(
            y_test_E_grad[:, 0:1], y_test_E_grad[:, 1:2], 
            test_points_req_grad, k0
        )
        
        with torch.no_grad():
            # Predict E field
            y_test_E = net(test_points_t)
            y_test = torch.cat([y_test_E, H_test_real.detach(), H_test_imag.detach()], dim=1)
            error_E = torch.sum(torch.abs(y_test[:, 0:2] - test_solution_t[:, 0:2])) / torch.sum(torch.abs(test_solution_t[:, 0:2]))
            error_H = torch.sum(torch.abs(y_test[:, 2:] - test_solution_t[:, 2:])) / torch.sum(torch.abs(test_solution_t[:, 2:]))
        print(f"E error: {error_E.item() * 100:.6e}% | H error: {error_H.item() * 100:.6e}%")
    print(f"Total time: {time.time() - start_time:.1f}s")
    
    # Save results
    if is_circle:
        save_results(net, loss_history, test_points, test_solution, geom)
    else:
        save_results(net, loss_history, None, None, geom)
    
    return net, loss_history

def save_results(net, loss_history, test_points, test_solution, geom):
    """Save training results and visualizations"""
    
    # Create output directory
    os.makedirs('results', exist_ok=True)
    
    # Save model
    torch.save(net.state_dict(), 'results/model.pt')
    print("Model saved to results/model.pt")
    
    # Plot loss history
    plt.figure(figsize=(15, 10))
    
    # Total loss
    plt.subplot(2, 3, 1)
    plt.semilogy(loss_history['total'], label='Total Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Total Loss')
    plt.legend()
    plt.grid(True)
    
    # PDE losses
    plt.subplot(2, 3, 2)
    plt.semilogy(loss_history['pde_0'], label='PDE Real')
    plt.semilogy(loss_history['pde_1'], label='PDE Imag')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('PDE Losses')
    plt.legend()
    plt.grid(True)
    
    # Inner BC losses
    plt.subplot(2, 3, 3)
    plt.semilogy(loss_history['bc_inner_0'], label='BC Inner Real')
    plt.semilogy(loss_history['bc_inner_1'], label='BC Inner Imag')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Inner Boundary Losses')
    plt.legend()
    plt.grid(True)
    
    # Outer BC losses
    plt.subplot(2, 3, 4)
    plt.semilogy(loss_history['bc_outer_0'], label='BC Outer Real')
    plt.semilogy(loss_history['bc_outer_1'], label='BC Outer Imag')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Outer Boundary Losses')
    plt.legend()
    plt.grid(True)
    
    # Data loss
    plt.subplot(2, 3, 5)
    if len(loss_history['data_loss']) > 0 and max(loss_history['data_loss']) > 0:
        plt.semilogy(loss_history['data_loss'], label='Data Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Data Loss')
    plt.legend()
    plt.grid(True)
    
    # Test error (only for circular geometry)
    plt.subplot(2, 3, 6)
    if len(loss_history['test_error_E']) > 0:
        iterations_test = np.arange(0, len(loss_history['total']), 100)[:len(loss_history['test_error'])]
        plt.semilogy(iterations_test, loss_history['test_error'], label='L2 Relative Error')
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.title('Test Error (vs Analytical)')
    else:
        plt.text(0.5, 0.5, 'No Analytical Solution', ha='center', va='center', fontsize=12)
        plt.xlabel('Iteration')
        plt.title('Test Error')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/loss_history.png', dpi=150)
    print("Loss history saved to results/loss_history.png")
    plt.close()
    
    # Visualize solution
    if test_points is not None:
        visualize_solution(net, test_points, test_solution, geom)

def visualize_solution(net, test_points, test_solution, geom):
    """Visualize predicted and exact solutions"""
    
    # Create fine grid for visualization
    n_viz = 100
    x_viz = np.linspace(-length/2, length/2, n_viz)
    y_viz = np.linspace(-length/2, length/2, n_viz)
    X_viz, Y_viz = np.meshgrid(x_viz, y_viz)
    points_viz = np.stack([X_viz.flatten(), Y_viz.flatten()], axis=1)

    inner_geom = geom.geom_b
    mask = np.logical_not(inner_geom.inside(points_viz))
    points_viz_filtered = points_viz[mask]
    
    # Get predictions for E field
    net.eval()
    
    points_viz_req_grad = torch.tensor(points_viz_filtered, dtype=torch.float32, device=device, requires_grad=True)
    E_pred_grad = net(points_viz_req_grad)
    H_pred_real_t, H_pred_imag_t = compute_H_from_E(
        E_pred_grad[:, 0:1], E_pred_grad[:, 1:2], 
        points_viz_req_grad, k0
    )
    
    with torch.no_grad():
        points_viz_t = torch.tensor(points_viz_filtered, dtype=torch.float32, device=device)
        E_pred_t = net(points_viz_t)
        
        # Convert to numpy
        E_pred_np = E_pred_t.cpu().numpy()
        H_pred_real_np = H_pred_real_t.detach().cpu().numpy().flatten()
        H_pred_imag_np = H_pred_imag_t.detach().cpu().numpy().flatten()
    
    # Get exact solution (4 components: E_real, E_imag, H_real, H_imag)
    y_exact = sol(points_viz_filtered)
    
    # Create full grids with NaN for interior points
    # E field
    E_pred_real = np.full(X_viz.shape, np.nan)
    E_pred_imag = np.full(X_viz.shape, np.nan)
    E_exact_real = np.full(X_viz.shape, np.nan)
    E_exact_imag = np.full(X_viz.shape, np.nan)
    # H field
    H_pred_real = np.full(X_viz.shape, np.nan)
    H_pred_imag = np.full(X_viz.shape, np.nan)
    H_exact_real = np.full(X_viz.shape, np.nan)
    H_exact_imag = np.full(X_viz.shape, np.nan)

    # Extract E field - predicted from network
    E_pred_real[mask.reshape(X_viz.shape)] = E_pred_np[:, 0]
    E_pred_imag[mask.reshape(X_viz.shape)] = E_pred_np[:, 1]
    # Extract E field - exact
    E_exact_real[mask.reshape(X_viz.shape)] = y_exact[:, 0]
    E_exact_imag[mask.reshape(X_viz.shape)] = y_exact[:, 1]
    
    # Extract H field - predicted from E field via Maxwell
    H_pred_real[mask.reshape(X_viz.shape)] = H_pred_real_np
    H_pred_imag[mask.reshape(X_viz.shape)] = H_pred_imag_np
    # Extract H field - exact
    H_exact_real[mask.reshape(X_viz.shape)] = y_exact[:, 2]
    H_exact_imag[mask.reshape(X_viz.shape)] = y_exact[:, 3]

    incident_wave_grid = np.exp(1j * k0 * X_viz)

    E_pred = (E_pred_real + 1j * E_pred_imag) + incident_wave_grid
    E_exact = (E_exact_real + 1j * E_exact_imag) + incident_wave_grid
    H_pred = (H_pred_real + 1j * H_pred_imag) + incident_wave_grid
    H_exact = (H_exact_real + 1j * H_exact_imag) + incident_wave_grid

    # Plot - visualize E field (since it's the total field with incident)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # E field - Predicted
    im0 = axes[0, 0].contourf(X_viz, Y_viz, np.abs(E_pred), levels=50, cmap='RdBu_r')
    axes[0, 0].set_title('E Field Predicted')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    plt.colorbar(im0, ax=axes[0, 0])
    _add_geometry_patch(axes[0, 0], inner_geom)
    axes[0, 0].set_aspect('equal')

    # E field - Exact
    im1 = axes[0, 1].contourf(X_viz, Y_viz, np.abs(E_exact), levels=50, cmap='RdBu_r')
    axes[0, 1].set_title('E Field Exact')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0, 1])
    _add_geometry_patch(axes[0, 1], inner_geom)
    axes[0, 1].set_aspect('equal')
    
    # E field - Error
    error_E = np.abs(E_pred - E_exact)
    im2 = axes[0, 2].contourf(X_viz, Y_viz, error_E, levels=50, cmap='viridis')
    axes[0, 2].set_title('E Field Absolute Error')
    axes[0, 2].set_xlabel('x')
    axes[0, 2].set_ylabel('y')
    plt.colorbar(im2, ax=axes[0, 2])
    _add_geometry_patch(axes[0, 2], inner_geom)
    axes[0, 2].set_aspect('equal')
    
    # H field - Predicted
    im3 = axes[1, 0].contourf(X_viz, Y_viz, np.abs(H_pred), levels=50, cmap='RdBu_r')
    axes[1, 0].set_title('H Field Predicted')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    plt.colorbar(im3, ax=axes[1, 0])
    _add_geometry_patch(axes[1, 0], inner_geom)
    axes[1, 0].set_aspect('equal')

    # H field - Exact
    im4 = axes[1, 1].contourf(X_viz, Y_viz, np.abs(H_exact), levels=50, cmap='RdBu_r')
    axes[1, 1].set_title('H Field Exact')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('y')
    plt.colorbar(im4, ax=axes[1, 1])
    _add_geometry_patch(axes[1, 1], inner_geom)
    axes[1, 1].set_aspect('equal')
    
    # H field - Error
    error_H = np.abs(H_pred - H_exact)
    im5 = axes[1, 2].contourf(X_viz, Y_viz, error_H, levels=50, cmap='viridis')
    axes[1, 2].set_title('H Field Absolute Error')
    axes[1, 2].set_xlabel('x')
    axes[1, 2].set_ylabel('y')
    plt.colorbar(im5, ax=axes[1, 2])
    _add_geometry_patch(axes[1, 2], inner_geom)
    axes[1, 2].set_aspect('equal')

    plt.tight_layout()
    plt.savefig('results/solution_comparison.png', dpi=150)
    print("Solution comparison saved to results/solution_comparison.png")
    plt.close()

def visualize_prediction_only(net, geom):
    """Visualize only predicted solution without analytical comparison"""
    
    # Create fine grid for visualization
    n_viz = 100
    x_viz = np.linspace(-length/2, length/2, n_viz)
    y_viz = np.linspace(-length/2, length/2, n_viz)
    X_viz, Y_viz = np.meshgrid(x_viz, y_viz)
    points_viz = np.stack([X_viz.flatten(), Y_viz.flatten()], axis=1)

    inner_geom = geom.geom_b
    outer_geom = geom.geom_a
    mask = np.logical_not(inner_geom.inside(points_viz))
    points_viz_filtered = points_viz[mask]
    
    # Get predictions
    net.eval()
    with torch.no_grad():
        points_viz_t = torch.tensor(points_viz_filtered, dtype=torch.float32, device=device)
        y_pred = net(points_viz_t).cpu().numpy()
    
    # Extract real and imaginary parts
    u_pred_real = y_pred[:, 0]
    u_pred_imag = y_pred[:, 1]
    
    # Save prediction data to txt file
    output_file = 'results/prediction_data.txt'
    with open(output_file, 'w') as f:
        f.write("# Predicted scattered field data\n")
        f.write("#" + "="*60 + "\n")
        f.write(get_geometry_info(outer_geom, "Outer Geometry"))
        f.write(get_geometry_info(inner_geom, "Inner Geometry (Scatterer)"))
        f.write("#" + "="*60 + "\n")
        f.write(f"# Wave number k0: {k0}\n")
        f.write(f"# Total points: {len(points_viz_filtered)}\n")
        f.write("# Format: x, y, real_part, imag_part\n")
        f.write("#" + "="*60 + "\n")
        for i in range(len(points_viz_filtered)):
            f.write(f"{points_viz_filtered[i, 0]:.6e}\t{points_viz_filtered[i, 1]:.6e}\t"
                   f"{u_pred_real[i]:.6e}\t{u_pred_imag[i]:.6e}\n")
    print(f"Prediction data saved ({len(points_viz_filtered)} points)")
    
    # Create full grids with NaN for interior points
    u_pred_real_grid = np.full(X_viz.shape, np.nan)
    u_pred_imag_grid = np.full(X_viz.shape, np.nan)

    u_pred_real_grid[mask.reshape(X_viz.shape)] = u_pred_real
    u_pred_imag_grid[mask.reshape(X_viz.shape)] = u_pred_imag

    # Compute total field (scattered + incident)
    incident_wave_grid = np.exp(1j * k0 * X_viz)
    u_total = (u_pred_real_grid + 1j * u_pred_imag_grid) + incident_wave_grid

    # Plot magnitude only
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    
    # Total field magnitude
    im = ax.contourf(X_viz, Y_viz, np.abs(u_total), levels=50, cmap='jet')
    ax.set_title('Total Field Magnitude', fontsize=16, fontweight='bold')
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('y', fontsize=14)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('|E|', fontsize=14)
    _add_geometry_patch(ax, inner_geom)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig('results/prediction_visualization.png', dpi=200, bbox_inches='tight')
    plt.close()


def plot_computational_domain_preview(outer, inner):
    """ outer  inner results/geometry_domain.png"""
    os.makedirs("results", exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    if isinstance(outer, Shape.RectangleGeometry):
        w = outer.xmax - outer.xmin
        h = outer.ymax - outer.ymin
        ax.add_patch(
            Rectangle(
                (outer.xmin, outer.ymin),
                w,
                h,
                facecolor="#e8f4fc",
                edgecolor="black",
                linewidth=1.5,
                zorder=1,
            )
        )
    else:
        _stroke_geometry_boundary(ax, outer)

    _add_geometry_patch(ax, inner)

    span = max(outer.xmax - outer.xmin, outer.ymax - outer.ymin)
    pad = 0.05 * span if span > 0 else 0.1
    ax.set_xlim(outer.xmin - pad, outer.xmax + pad)
    ax.set_ylim(outer.ymin - pad, outer.ymax + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Computational domain (outer) and scatterer (inner)")
    ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig("results/geometry_domain.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Geometry preview saved to results/geometry_domain.png")


def _add_geometry_patch(ax, geometry):
    """patch"""
    
    if isinstance(geometry, Shape.RectangleGeometry):
        width = geometry.xmax - geometry.xmin
        height = geometry.ymax - geometry.ymin
        rect = Rectangle((geometry.xmin, geometry.ymin), width, height, 
                        color='white', fill=True, edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
    
    elif isinstance(geometry, Shape.Disk):
        circle = Circle(geometry.center, geometry.radius, 
                       color='white', fill=True, edgecolor='black', linewidth=1.5)
        ax.add_patch(circle)
    
    elif isinstance(geometry, Shape.Ellipse):
        ellipse = MplEllipse(geometry.center, 2*geometry.semimajor, 2*geometry.semiminor,
                           angle=geometry.angle,
                           color='white', fill=True, edgecolor='black', linewidth=1.5)
        ax.add_patch(ellipse)
    
    elif isinstance(geometry, (Shape.Polygon, Shape.StarShaped)):
        vertices_closed = np.vstack([geometry.vertices, geometry.vertices[0]])
        poly = MplPolygon(vertices_closed, 
                         color='white', fill=True, edgecolor='black', linewidth=1.5, 
                         closed=True)
        ax.add_patch(poly)

    elif isinstance(geometry, Shape.GeometryDifference):
        _stroke_geometry_boundary(ax, geometry.geom_a)
        _stroke_geometry_boundary(ax, geometry.geom_b)

    elif isinstance(geometry, (Shape.SuperEllipse, Shape.RadialFourier)):
        bd = getattr(geometry, "boundary_pts", None)
        if bd is not None and len(bd) > 2:
            poly = MplPolygon(
                bd,
                color="white",
                fill=True,
                edgecolor="black",
                linewidth=1.5,
                closed=True,
            )
            ax.add_patch(poly)
        else:
            _stroke_geometry_boundary(ax, geometry)

    else:
        try:
            boundary_pts = geometry.random_boundary_points(100)
            ax.plot(boundary_pts[:, 0], boundary_pts[:, 1], 'k-', linewidth=1.5)
        except:
            print(f"Warning: Cannot visualize geometry type {type(geometry).__name__}")

def load_and_visualize():
    """Load existing model and visualize results"""
    print("Loading existing model for visualization...")
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    outer = Shape.RectangleGeometry(-length / 2, length / 2, -length / 2, length / 2)
    # inner = Shape.RectangleGeometry(-1, 1, -0.1197, 0.1197)  # IQ = 0.3
    # inner = Shape.scatterer_crescent(R_big=1.0, R_small=0.55, offset=0.35)
    # inner = Shape.scatterer_elliptical_ring([0, 0], 1.0, 0.5, inner_scale=0.5)
    # inner = Shape.polygon_stadium(half_length=0.7, radius=0.35)

    #
    #
    # inner = Shape.polygon_T_shape(stem_half_width=0.12, stem_height=0.90,
    # inner = Shape.polygon_arrow(body_half_width=0.12, body_length=0.70,

    inner = Shape.polygon_airplane_profile(scale=1.0)
    #
    #
    #
    # -------------------------------------------------------------------------
    # vertices = [
    #     [0.000000, -0.618034],
    #     [-0.138757, -0.190983],
    #     [-0.587785, -0.190983],
    #     [-0.224514, 0.072949],
    #     [-0.363271, 0.500000],
    #     [0.000000, 0.236068],
    #     [0.363271, 0.500000],
    #     [0.224514, 0.072949],
    #     [0.587785, -0.190983],
    #     [0.138757, -0.190983]
    # ] # 0.8450, 0.5279

    geom = Shape.GeometryDifference(outer, inner)
    
    net = rbf_net.PINN_module(n_in, n_out, n_neu_x, n_neu_y, b, c_x, c_y, device).to(device)
    
    # Load model parameters
    model_path = 'results/model.pt'

    net.load_state_dict(torch.load(model_path, map_location=device))
    
    visualize_prediction_only(net, geom)
    
    return net

if __name__ == '__main__':
    print("=" * 80)
    print("Pure PyTorch Helmholtz Scattering Solver")
    print("=" * 80)
    
    if TRAIN_MODEL:
        print("\nTraining model...")
        net, history = train_model()
    else:
        print("\nLoading model and visualizing...")
        net = load_and_visualize()

    # 
    #    outer = Shape.RectangleGeometry(-length/2, length/2, -length/2, length/2)
    #    geom = Shape.GeometryDifference(outer, inner)
    #
    #    geom = Shape.GeometryDifference(outer, inner)
    #
    #    vertices = [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5], [0, 0.8]]
    #    outer = Shape.RectangleGeometry(-length/2, length/2, -length/2, length/2)
    #    inner = Shape.Polygon(vertices)
    #    geom = Shape.GeometryDifference(outer, inner)
    #
    #    outer = Shape.RectangleGeometry(-length/2, length/2, -length/2, length/2)
    #    inner = Shape.StarShaped([0, 0], num_points=5, inner_radius=0.3, outer_radius=0.6)
    #    geom = Shape.GeometryDifference(outer, inner)
    #
    #    outer = Shape.RectangleGeometry(-length/2, length/2, -length/2, length/2)
    #    inner = Shape.Disk([0, 0], radius=R)
    #    geom = Shape.GeometryDifference(outer, inner)
    #
    #    outer = Shape.RectangleGeometry(-length/2, length/2, -length/2, length/2)
    #    inner = Shape.RectangleGeometry(-L/2, L/2, -L/2, L/2)
    #    geom = Shape.GeometryDifference(outer, inner)
    #
    #    inner = Shape.scatterer_crescent(R_big=1.0, R_small=0.55, offset=0.35)
    #    inner = Shape.scatterer_elliptical_ring([0,0], 1.0, 0.5, inner_scale=0.5)
    #    inner = Shape.polygon_stadium(half_length=0.7, radius=0.35)
    #    inner = Shape.polygon_cross(arm_half_length=0.9, arm_half_width=0.22)
    #    geom = Shape.GeometryDifference(outer, inner)
