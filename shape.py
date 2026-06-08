import numpy as np
import torch
from scipy.special import jv, hankel1

class Shape:
    class RectangleGeometry:
        """Rectangle geometry"""
        def __init__(self, xmin, xmax, ymin, ymax):
            self.xmin = xmin
            self.xmax = xmax
            self.ymin = ymin
            self.ymax = ymax
            self.bbox = np.array([[xmin, ymin], [xmax, ymax]])

        def inside(self, x):
            """Check if points are inside rectangle (including boundary)"""
            return np.logical_and.reduce([
                x[:, 0] >= self.xmin,
                x[:, 0] <= self.xmax,
                x[:, 1] >= self.ymin,
                x[:, 1] <= self.ymax
            ])

        def on_boundary(self, x):
            """Check if points are on rectangle boundary"""
            x_on = np.logical_or(
                np.isclose(x[:, 0], self.xmin),
                np.isclose(x[:, 0], self.xmax)
            )
            y_on = np.logical_or(
                np.isclose(x[:, 1], self.ymin),
                np.isclose(x[:, 1], self.ymax)
            )
            inside_x = np.logical_and(x[:, 0] >= self.xmin, x[:, 0] <= self.xmax)
            inside_y = np.logical_and(x[:, 1] >= self.ymin, x[:, 1] <= self.ymax)

            return np.logical_and(np.logical_or(x_on, y_on), 
                                 np.logical_and(inside_x, inside_y))

        def boundary_normal(self, x):
            """Compute outward unit normal at boundary points"""
            normals = np.zeros_like(x)

            for i, point in enumerate(x):
                # Determine which edge the point is on
                if np.isclose(point[0], self.xmin):
                    normals[i] = [-1, 0]  # Left edge, normal points left
                elif np.isclose(point[0], self.xmax):
                    normals[i] = [1, 0]  # Right edge, normal points right
                elif np.isclose(point[1], self.ymin):
                    normals[i] = [0, -1]  # Bottom edge, normal points down
                elif np.isclose(point[1], self.ymax):
                    normals[i] = [0, 1]  # Top edge, normal points up
                else:
                    # Point is not exactly on boundary, use closest boundary
                    dist_to_left = abs(point[0] - self.xmin)
                    dist_to_right = abs(point[0] - self.xmax)
                    dist_to_bottom = abs(point[1] - self.ymin)
                    dist_to_top = abs(point[1] - self.ymax)

                    min_dist = min(dist_to_left, dist_to_right, dist_to_bottom, dist_to_top)

                    if np.isclose(min_dist, dist_to_left):
                        normals[i] = [-1, 0]
                    elif np.isclose(min_dist, dist_to_right):
                        normals[i] = [1, 0]
                    elif np.isclose(min_dist, dist_to_bottom):
                        normals[i] = [0, -1]
                    elif np.isclose(min_dist, dist_to_top):
                        normals[i] = [0, 1]

            return normals

        def random_points(self, n, sampler='Hammersley'):
            """Generate random points inside rectangle"""
            if sampler == 'Hammersley':
                # Use Hammersley sequence
                try:
                    import skopt
                    sampler_obj = skopt.sampler.Hammersly(min_skip=-1, max_skip=-1)
                    space = [(0.0, 1.0), (0.0, 1.0)]
                    samples = np.array(sampler_obj.generate(space, n))
                except:
                    # Fallback to pseudorandom
                    samples = np.random.rand(n, 2)
            else:
                samples = np.random.rand(n, 2)

            # Map from [0,1]^2 to rectangle
            points = np.zeros((n, 2))
            points[:, 0] = self.xmin + samples[:, 0] * (self.xmax - self.xmin)
            points[:, 1] = self.ymin + samples[:, 1] * (self.ymax - self.ymin)
            return points

        def random_boundary_points(self, n):
            """Generate random points on rectangle boundary"""
            # Distribute points on 4 edges proportional to their length
            perimeter = 2 * (self.xmax - self.xmin) + 2 * (self.ymax - self.ymin)
            n_bottom = int(n * (self.xmax - self.xmin) / perimeter)
            n_top = int(n * (self.xmax - self.xmin) / perimeter)
            n_left = int(n * (self.ymax - self.ymin) / perimeter)
            n_right = n - n_bottom - n_top - n_left

            points = []
            # Bottom edge
            t = np.random.rand(n_bottom)
            points.append(np.stack([self.xmin + t * (self.xmax - self.xmin), 
                                   np.full(n_bottom, self.ymin)], axis=1))
            # Top edge
            t = np.random.rand(n_top)
            points.append(np.stack([self.xmin + t * (self.xmax - self.xmin), 
                                   np.full(n_top, self.ymax)], axis=1))
            # Left edge
            t = np.random.rand(n_left)
            points.append(np.stack([np.full(n_left, self.xmin), 
                                   self.ymin + t * (self.ymax - self.ymin)], axis=1))
            # Right edge
            t = np.random.rand(n_right)
            points.append(np.stack([np.full(n_right, self.xmax), 
                                   self.ymin + t * (self.ymax - self.ymin)], axis=1))

            return np.vstack(points)


        @property
        def area(self):
            return (self.xmax - self.xmin) * (self.ymax - self.ymin)

        @property
        def perimeter(self):
            return 2 * ((self.xmax - self.xmin) + (self.ymax - self.ymin))

        @property
        def IQ(self):
            return 4 * np.pi * self.area / self.perimeter**2

    class Disk:
        """Disk (circle) geometry"""
        def __init__(self, center, radius):
            self.center = np.array(center)
            self.radius = radius

        def inside(self, x):
            """Check if points are inside disk (including boundary)"""
            r = np.linalg.norm(x - self.center, axis=1)
            return r <= self.radius

        def on_boundary(self, x):
            """Check if points are on disk boundary"""
            r = np.linalg.norm(x - self.center, axis=1)
            return np.isclose(r, self.radius)

        def boundary_normal(self, x):
            """Compute outward unit normal at boundary points"""
            vec = x - self.center
            norm = np.linalg.norm(vec, axis=1, keepdims=True)
            norm = np.where(norm == 0, 1.0, norm)  # Avoid division by zero
            return vec / norm

        def random_points(self, n):
            """Generate random points inside disk"""
            # Use rejection sampling for simplicity
            points = []
            while len(points) < n:
                candidates = np.random.rand(n * 2, 2) * 2 * self.radius - self.radius
                candidates = candidates + self.center
                mask = self.inside(candidates)
                points.extend(candidates[mask])
            return np.array(points[:n])

        def random_boundary_points(self, n):
            """Generate random points on disk boundary"""
            theta = np.random.rand(n) * 2 * np.pi
            points = np.zeros((n, 2))
            points[:, 0] = self.center[0] + self.radius * np.cos(theta)
            points[:, 1] = self.center[1] + self.radius * np.sin(theta)
            return points


        @property
        def area(self):
            return np.pi * (self.radius**2)

        @property
        def perimeter(self):
            return 2 * np.pi * self.radius

        @property
        def IQ(self):
            return 4 * np.pi * self.area / self.perimeter**2

    class Ellipse:
        """Ellipse geometry with rotation support"""
        def __init__(self, center, semimajor, semiminor, angle=0):
            """
            Args:
                center: [x, y] center coordinates
                semimajor: semi-major axis (a)
                semiminor: semi-minor axis (b)
                angle: rotation angle in degrees (default: 0)
                       positive angle rotates counter-clockwise
            """
            self.center = np.array(center)
            self.semimajor = semimajor
            self.semiminor = semiminor
            self.axis_radius = np.array([semimajor, semiminor])
            self.angle = angle  # in degrees
            self.angle_rad = np.radians(angle)  # in radians

            # Pre-compute rotation matrices
            cos_a = np.cos(self.angle_rad)
            sin_a = np.sin(self.angle_rad)
            # Rotation matrix: global -> local (inverse rotation)
            self.rot_matrix_inv = np.array([
                [cos_a, sin_a],
                [-sin_a, cos_a]
            ])
            # Rotation matrix: local -> global
            self.rot_matrix = np.array([
                [cos_a, -sin_a],
                [sin_a, cos_a]
            ])

        def _to_local(self, x):
            """Transform points from global to local coordinate system"""
            diff = x - self.center
            return diff @ self.rot_matrix_inv.T

        def _to_global(self, x_local):
            """Transform points from local to global coordinate system"""
            return x_local @ self.rot_matrix.T + self.center

        def inside(self, x):
            """Check if points are inside ellipse (including boundary)"""
            x_local = self._to_local(x)
            return np.linalg.norm(x_local / self.axis_radius, axis=-1) <= 1

        def on_boundary(self, x):
            """Check if points are on ellipse boundary"""
            x_local = self._to_local(x)
            return np.isclose(np.linalg.norm(x_local / self.axis_radius, axis=-1), 1)

        def boundary_normal(self, x):
            """Compute outward unit normal at boundary points"""
            # Transform to local coordinates
            x_local = self._to_local(x)

            # For ellipse in local coordinates: x^2/a^2 + y^2/b^2 = 1
            # Normal direction in local: [x/a^2, y/b^2] or [b^2*x, a^2*y]
            normals_local = np.zeros_like(x_local)
            normals_local[:, 0] = self.semiminor**2 * x_local[:, 0]
            normals_local[:, 1] = self.semimajor**2 * x_local[:, 1]

            # Normalize in local coordinates
            norm = np.linalg.norm(normals_local, axis=1, keepdims=True)
            norm = np.where(norm == 0, 1.0, norm)
            normals_local = normals_local / norm

            # Transform normals back to global coordinates
            normals_global = normals_local @ self.rot_matrix.T

            # Only return normals for points on boundary
            on_boundary_mask = self.on_boundary(x)
            normals_global = normals_global * on_boundary_mask[:, np.newaxis]

            return normals_global

        def random_points(self, n, sampler='pseudo'):
            """Generate random points inside ellipse"""
            # Generate points in local coordinate system
            if sampler == 'Hammersley':
                try:
                    import skopt
                    sampler_obj = skopt.sampler.Hammersly(min_skip=-1, max_skip=-1)
                    space = [(0.0, 1.0), (0.0, 1.0)]
                    rng = np.array(sampler_obj.generate(space, n))
                except:
                    rng = np.random.rand(n, 2)
            else:
                rng = np.random.rand(n, 2)

            r, theta = rng[:, 0], 2 * np.pi * rng[:, 1]
            x, y = np.cos(theta), np.sin(theta)
            points_local = self.axis_radius * (np.sqrt(r) * np.vstack((x, y))).T

            # Transform to global coordinates
            points_global = self._to_global(points_local)
            return points_global

        def random_boundary_points(self, n):
            """Generate random points on ellipse boundary"""
            theta = np.random.rand(n) * 2 * np.pi

            # Generate points in local coordinate system
            points_local = np.zeros((n, 2))
            points_local[:, 0] = self.semimajor * np.cos(theta)
            points_local[:, 1] = self.semiminor * np.sin(theta)

            # Transform to global coordinates
            points_global = self._to_global(points_local)
            return points_global


        @property
        def area(self):
            return np.pi * self.semimajor * self.semiminor

        @property
        def perimeter(self):
            from scipy import integrate
            def ds(t):
                dx = -self.semimajor * np.sin(t)
                dy = self.semiminor * np.cos(t)
                return np.sqrt(dx**2 + dy**2)
            p, _ = integrate.quad(ds, 0, 2 * np.pi)
            return p

        @property
        def IQ(self):
            return 4 * np.pi * self.area / self.perimeter**2

    class Polygon:
        """Polygon geometry (simple polygon without self-intersection)"""
        def __init__(self, vertices):
            """
            Args:
                vertices: List of [x, y] coordinates of polygon vertices
                         Order can be clockwise or counterclockwise
            """
            self.vertices = np.array(vertices, dtype=np.float64)
            self.nvertices = len(self.vertices)

            # Compute signed area to determine orientation
            self.area = self._polygon_signed_area()

            # Ensure counterclockwise orientation
            if self.area < 0:
                self.area = -self.area
                self.vertices = np.flipud(self.vertices)

            # Compute edge information
            self.segments = np.roll(self.vertices, -1, axis=0) - self.vertices
            self.edge_lengths = np.linalg.norm(self.segments, axis=1)
            self.perimeter = np.sum(self.edge_lengths)

            # Compute outward normals (90 degree clockwise rotation of edges)
            self.normals = np.zeros_like(self.segments)
            self.normals[:, 0] = self.segments[:, 1]
            self.normals[:, 1] = -self.segments[:, 0]
            self.normals = self.normals / self.edge_lengths[:, np.newaxis]

            # Bounding box
            self.bbox = np.array([np.min(self.vertices, axis=0), 
                                 np.max(self.vertices, axis=0)])

        def _polygon_signed_area(self):
            """Compute signed area using shoelace formula"""
            x = self.vertices[:, 0]
            y = self.vertices[:, 1]
            return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

        def inside(self, x):
            """Check if points are inside polygon using winding number algorithm"""
            def winding_number(point):
                wn = 0
                for i in range(self.nvertices):
                    v0 = self.vertices[i]
                    v1 = self.vertices[(i + 1) % self.nvertices]

                    if v0[1] <= point[1]:
                        if v1[1] > point[1]:  # Upward crossing
                            if self._is_left(v0, v1, point) > 0:
                                wn += 1
                    else:
                        if v1[1] <= point[1]:  # Downward crossing
                            if self._is_left(v0, v1, point) < 0:
                                wn -= 1
                return wn != 0

            return np.array([winding_number(point) for point in x])

        def _is_left(self, P0, P1, P2):
            """Test if point P2 is left of line P0->P1"""
            return (P1[0] - P0[0]) * (P2[1] - P0[1]) - (P2[0] - P0[0]) * (P1[1] - P0[1])

        def on_boundary(self, x):
            """Check if points are on polygon boundary"""
            on_bdry = np.zeros(len(x), dtype=bool)

            for i in range(self.nvertices):
                v0 = self.vertices[i]
                v1 = self.vertices[(i + 1) % self.nvertices]

                # Distance from point to edge endpoints
                d0 = np.linalg.norm(x - v0, axis=1)
                d1 = np.linalg.norm(x - v1, axis=1)
                edge_len = self.edge_lengths[i]

                # Point is on edge if d0 + d1  edge_length
                on_bdry |= np.isclose(d0 + d1, edge_len)

            return on_bdry

        def boundary_normal(self, x):
            """Compute outward unit normal at boundary points"""
            normals = np.zeros_like(x)

            for i, point in enumerate(x):
                # Find which edge the point is on
                for j in range(self.nvertices):
                    v0 = self.vertices[j]
                    v1 = self.vertices[(j + 1) % self.nvertices]

                    d0 = np.linalg.norm(point - v0)
                    d1 = np.linalg.norm(point - v1)

                    if np.isclose(d0 + d1, self.edge_lengths[j]):
                        normals[i] = self.normals[j]
                        break

            return normals

        def random_points(self, n, sampler='Hammersley'):
            """Generate random points inside polygon using rejection sampling"""
            points = []
            vbbox = self.bbox[1] - self.bbox[0]

            while len(points) < n:
                # Generate candidates in bounding box
                if sampler == 'Hammersley' and len(points) == 0:
                    try:
                        import skopt
                        sampler_obj = skopt.sampler.Hammersly(min_skip=-1, max_skip=-1)
                        space = [(0.0, 1.0), (0.0, 1.0)]
                        samples = np.array(sampler_obj.generate(space, n * 2))
                        candidates = samples * vbbox + self.bbox[0]
                    except:
                        candidates = np.random.rand(n * 2, 2) * vbbox + self.bbox[0]
                else:
                    candidates = np.random.rand(n * 2, 2) * vbbox + self.bbox[0]

                # Filter points inside polygon
                mask = self.inside(candidates)
                points.extend(candidates[mask])

            return np.array(points[:n])

        def random_boundary_points(self, n):
            """Generate random points on polygon boundary"""
            # Distribute points according to edge lengths
            u = np.random.rand(n) * self.perimeter

            points = []
            for i in range(n):
                # Find which edge this point belongs to
                cumulative_length = 0
                for j in range(self.nvertices):
                    if u[i] <= cumulative_length + self.edge_lengths[j]:
                        # Interpolate along edge j
                        t = (u[i] - cumulative_length) / self.edge_lengths[j]
                        v0 = self.vertices[j]
                        v1 = self.vertices[(j + 1) % self.nvertices]
                        point = v0 + t * (v1 - v0)
                        points.append(point)
                        break
                    cumulative_length += self.edge_lengths[j]

            return np.array(points)

        @property
        def IQ(self):
            return 4 * np.pi * self.area / self.perimeter**2

    class StarShaped:
        """Star-shaped geometry defined by radius function r(theta)"""
        def __init__(self, center, num_points, inner_radius, outer_radius):
            """
            Args:
                center: [x, y] center coordinates
                num_points: number of star points
                inner_radius: radius at valleys
                outer_radius: radius at peaks
            """
            self.center = np.array(center)
            self.num_points = num_points
            self.inner_radius = inner_radius
            self.outer_radius = outer_radius

            # Pre-compute star vertices for boundary operations
            self.vertices = self._compute_vertices()
            self.nvertices = len(self.vertices)

            # Use polygon-like edge computation
            self.segments = np.roll(self.vertices, -1, axis=0) - self.vertices
            self.edge_lengths = np.linalg.norm(self.segments, axis=1)
            self.perimeter = np.sum(self.edge_lengths)

            # Compute normals
            self.normals = np.zeros_like(self.segments)
            self.normals[:, 0] = self.segments[:, 1]
            self.normals[:, 1] = -self.segments[:, 0]
            self.normals = self.normals / self.edge_lengths[:, np.newaxis]

        def _compute_vertices(self):
            """Compute star vertices (alternating between peaks and valleys)"""
            vertices = []
            for i in range(2 * self.num_points):
                theta = i * np.pi / self.num_points
                if i % 2 == 0:  # Peak
                    r = self.outer_radius
                else:  # Valley
                    r = self.inner_radius
                x = self.center[0] + r * np.cos(theta)
                y = self.center[1] + r * np.sin(theta)
                vertices.append([x, y])
            return np.array(vertices)

        def _radius_at_angle(self, theta):
            """Get radius at given angle"""
            # Normalize angle to [0, 2)
            theta = np.mod(theta, 2 * np.pi)

            # Each point spans an angle of 2 / num_points
            segment_angle = 2 * np.pi / self.num_points
            segment_idx = (theta // (segment_angle / 2)).astype(int)

            # Alternate between outer and inner radius
            r = np.where(segment_idx % 2 == 0, self.outer_radius, self.inner_radius)
            return r

        def inside(self, x):
            """Check if points are inside star"""
            diff = x - self.center
            r = np.linalg.norm(diff, axis=1)
            theta = np.arctan2(diff[:, 1], diff[:, 0])
            r_boundary = self._radius_at_angle(theta)
            return r <= r_boundary

        def on_boundary(self, x):
            """Check if points are on star boundary"""
            # Check if close to any vertex
            for vertex in self.vertices:
                if np.any(np.linalg.norm(x - vertex, axis=1) < 1e-6):
                    return np.linalg.norm(x - vertex, axis=1) < 1e-6

            # Check if on edges
            on_bdry = np.zeros(len(x), dtype=bool)
            for i in range(self.nvertices):
                v0 = self.vertices[i]
                v1 = self.vertices[(i + 1) % self.nvertices]
                d0 = np.linalg.norm(x - v0, axis=1)
                d1 = np.linalg.norm(x - v1, axis=1)
                edge_len = self.edge_lengths[i]
                on_bdry |= np.isclose(d0 + d1, edge_len)

            return on_bdry

        def boundary_normal(self, x):
            """Compute outward unit normal at boundary points"""
            normals = np.zeros_like(x)

            for i, point in enumerate(x):
                # Find which edge the point is on
                for j in range(self.nvertices):
                    v0 = self.vertices[j]
                    v1 = self.vertices[(j + 1) % self.nvertices]

                    d0 = np.linalg.norm(point - v0)
                    d1 = np.linalg.norm(point - v1)

                    if np.isclose(d0 + d1, self.edge_lengths[j]):
                        normals[i] = self.normals[j]
                        break

            return normals

        def random_points(self, n, sampler='pseudo'):
            """Generate random points inside star using rejection sampling"""
            points = []
            # Use bounding circle
            max_radius = self.outer_radius

            while len(points) < n:
                if sampler == 'Hammersley' and len(points) == 0:
                    try:
                        import skopt
                        sampler_obj = skopt.sampler.Hammersly(min_skip=-1, max_skip=-1)
                        space = [(0.0, 1.0), (0.0, 1.0)]
                        rng = np.array(sampler_obj.generate(space, n * 2))
                    except:
                        rng = np.random.rand(n * 2, 2)
                else:
                    rng = np.random.rand(n * 2, 2)

                # Generate in polar coordinates
                r = np.sqrt(rng[:, 0]) * max_radius
                theta = rng[:, 1] * 2 * np.pi

                candidates = np.zeros((len(r), 2))
                candidates[:, 0] = self.center[0] + r * np.cos(theta)
                candidates[:, 1] = self.center[1] + r * np.sin(theta)

                # Filter points inside star
                mask = self.inside(candidates)
                points.extend(candidates[mask])

            return np.array(points[:n])

        def random_boundary_points(self, n):
            """Generate random points on star boundary"""
            # Distribute points according to perimeter
            u = np.random.rand(n) * self.perimeter

            points = []
            for i in range(n):
                cumulative_length = 0
                for j in range(self.nvertices):
                    if u[i] <= cumulative_length + self.edge_lengths[j]:
                        t = (u[i] - cumulative_length) / self.edge_lengths[j]
                        v0 = self.vertices[j]
                        v1 = self.vertices[(j + 1) % self.nvertices]
                        point = v0 + t * (v1 - v0)
                        points.append(point)
                        break
                    cumulative_length += self.edge_lengths[j]

            return np.array(points)


        @property
        def area(self):
            x = self.vertices[:, 0]
            y = self.vertices[:, 1]
            return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

        @property
        def IQ(self):
            return 4 * np.pi * self.area / self.perimeter**2

    class GeometryDifference:
        """Geometry representing A - B (A minus B)"""
        def __init__(self, geom_a, geom_b):
            self.geom_a = geom_a
            self.geom_b = geom_b

        def inside(self, x):
            """Points inside A but not inside B"""
            return np.logical_and(self.geom_a.inside(x), 
                                 np.logical_not(self.geom_b.inside(x)))

        def on_boundary(self, x):
            """Points on boundary of A or on boundary of B (if inside A)"""
            on_a_boundary = np.logical_and(self.geom_a.on_boundary(x),
                                           np.logical_not(self.geom_b.inside(x)))
            on_b_boundary = np.logical_and(self.geom_b.on_boundary(x),
                                           self.geom_a.inside(x))
            return np.logical_or(on_a_boundary, on_b_boundary)

        def random_points(self, n, sampler='Hammersley'):
            """Generate random points in A - B"""
            # Generate points in A and filter out those in B
            # Generate extra points to account for filtering
            n_samples = int(n * 1.5)

            def _rp_a(ns, sam):
                try:
                    return self.geom_a.random_points(ns, sam)
                except TypeError:
                    return self.geom_a.random_points(ns)

            points_a = _rp_a(n_samples, sampler)
            # Filter out points inside B
            mask = np.logical_not(self.geom_b.inside(points_a))
            points = points_a[mask]
            # If we don't have enough, generate more
            while len(points) < n:
                more_points = _rp_a(n_samples, "pseudo")
                mask = np.logical_not(self.geom_b.inside(more_points))
                points = np.vstack([points, more_points[mask]])

            return points

        def random_boundary_points(self, n):
            """Generate random points on boundary of A - B"""
            # Split between outer boundary (A) and inner boundary (B)
            n_outer = n//2
            n_inner = n - n_outer

            # Outer boundary: points on A that are not in B
            outer_points = self.geom_a.random_boundary_points(n_outer * 2)

            mask = np.logical_not(self.geom_b.inside(outer_points))
            outer_points = outer_points[mask]

            # Inner boundary: points on B
            inner_points = self.geom_b.random_boundary_points(n_inner)

            return np.vstack([outer_points, inner_points])

        def boundary_normal(self, x):
            """Compute boundary normal (outward from domain)"""
            normals = np.zeros_like(x)

            # For points on outer boundary (A), use A's normal
            on_a = self.geom_a.on_boundary(x)

            # Handle different geometry types for outer boundary
            if isinstance(self.geom_a, Shape.RectangleGeometry):
                for i, point in enumerate(x):
                    if not on_a[i]:
                        continue
                    # Determine which edge
                    if np.isclose(point[0], self.geom_a.xmin):
                        normals[i] = [-1, 0]
                    elif np.isclose(point[0], self.geom_a.xmax):
                        normals[i] = [1, 0]
                    elif np.isclose(point[1], self.geom_a.ymin):
                        normals[i] = [0, -1]
                    elif np.isclose(point[1], self.geom_a.ymax):
                        normals[i] = [0, 1]
            elif isinstance(self.geom_a, (Shape.Disk, Shape.Ellipse, Shape.Polygon, Shape.StarShaped)):
                # These geometries have their own boundary_normal methods
                a_normals = self.geom_a.boundary_normal(x)
                normals[on_a] = a_normals[on_a]

            # For points on inner boundary (B), use negative of B's normal (pointing into domain)
            on_b = self.geom_b.on_boundary(x)

            # Handle different geometry types for inner boundary
            if isinstance(self.geom_b, Shape.Disk):
                b_normals = self.geom_b.boundary_normal(x)
                normals[on_b] = -b_normals[on_b]
            elif isinstance(self.geom_b, Shape.RectangleGeometry):
                for i, point in enumerate(x):
                    if not on_b[i]:
                        continue
                    if np.isclose(point[0], self.geom_b.xmin):
                        normals[i] = [1, 0]
                    elif np.isclose(point[0], self.geom_b.xmax):
                        normals[i] = [-1, 0]
                    elif np.isclose(point[1], self.geom_b.ymin):
                        normals[i] = [0, 1]
                    elif np.isclose(point[1], self.geom_b.ymax):
                        normals[i] = [0, -1]
            elif isinstance(self.geom_b, (Shape.Ellipse, Shape.Polygon, Shape.StarShaped)):
                # These geometries have their own boundary_normal methods
                # Negate the normal to point into the domain
                b_normals = self.geom_b.boundary_normal(x)
                normals[on_b] = -b_normals[on_b]

            return normals

    # ==================== Sampling Functions ====================

    class SuperEllipse:
        def __init__(self, center, a, b, n, angle=0):
            self.center = np.array(center)
            self.a = a
            self.b = b
            self.n = n
            self.angle = angle
            self.bbox = [self.center - np.array([a, b]), self.center + np.array([a, b])]

            t = np.linspace(0, 2*np.pi, 2000)
            xt = a * np.sign(np.cos(t)) * np.abs(np.cos(t))**(2/n)
            yt = b * np.sign(np.sin(t)) * np.abs(np.sin(t))**(2/n)
            if angle != 0:
                rad = np.radians(angle)
                cos_a, sin_a = np.cos(rad), np.sin(rad)
                xt_rot = xt * cos_a - yt * sin_a
                yt_rot = xt * sin_a + yt * cos_a
                xt, yt = xt_rot, yt_rot
            self.boundary_pts = np.vstack([self.center[0] + xt, self.center[1] + yt]).T

        def inside(self, x):
            dx = x[:, 0] - self.center[0]
            dy = x[:, 1] - self.center[1]
            if self.angle != 0:
                rad = np.radians(self.angle)
                cos_a, sin_a = np.cos(rad), np.sin(rad)
                dx_rot = dx * cos_a + dy * sin_a
                dy_rot = -dx * sin_a + dy * cos_a
                dx, dy = dx_rot, dy_rot
            eps = 1e-12
            val = (np.abs(dx / self.a)**self.n + np.abs(dy / self.b)**self.n)
            return val <= 1.0 + eps

        def on_boundary(self, x):
            from scipy.spatial import cKDTree
            tree = cKDTree(self.boundary_pts)
            dist, _ = tree.query(x)
            return dist < 1e-2

        def boundary_normal(self, x):
            dx = x[:, 0] - self.center[0]
            dy = x[:, 1] - self.center[1]
            if self.angle != 0:
                rad = np.radians(self.angle)
                cos_a, sin_a = np.cos(rad), np.sin(rad)
                dx, dy = dx * cos_a + dy * sin_a, -dx * sin_a + dy * cos_a
            
            # df/dx = n/a * sgn(x) * |x/a|^(n-1)
            eps = 1e-10
            df_dx = (self.n / self.a) * np.sign(dx) * (np.abs(dx / self.a + eps))**(self.n - 1)
            df_dy = (self.n / self.b) * np.sign(dy) * (np.abs(dy / self.b + eps))**(self.n - 1)
            
            # Rotate normals back to original frame
            if self.angle != 0:
                rad = np.radians(self.angle)
                cos_a, sin_a = np.cos(rad), np.sin(rad)
                df_dx_rot = df_dx * cos_a - df_dy * sin_a
                df_dy_rot = df_dx * sin_a + df_dy * cos_a
                df_dx, df_dy = df_dx_rot, df_dy_rot

            grad = np.vstack([df_dx, df_dy]).T
            norm = np.linalg.norm(grad, axis=1, keepdims=True)
            return grad / (norm + 1e-12)

        def random_boundary_points(self, n):
            idx = np.random.choice(self.boundary_pts.shape[0], n, replace=False)
            return self.boundary_pts[idx]

        @property
        def area(self):
            import scipy.special as sp
            return 4 * self.a * self.b * (sp.gamma(1 + 1/self.n)**2) / sp.gamma(1 + 2/self.n)

        @property
        def perimeter(self):
            diff = np.diff(self.boundary_pts, axis=0)
            p = np.sum(np.linalg.norm(diff, axis=1))
            p += np.linalg.norm(self.boundary_pts[-1] - self.boundary_pts[0])
            return p

        @property
        def IQ(self):
            return 4 * np.pi * self.area / self.perimeter**2

    class RadialFourier:
        def __init__(self, center, r0, a, m, angle=0):
            # r(t) = r0 + a*cos(m*t)
            self.center = np.array(center)
            self.r0 = r0
            self.a = a
            self.m = m
            self.angle = angle
            self.bbox = [self.center - (r0+a), self.center + (r0+a)]

            t = np.linspace(0, 2*np.pi, 2000)
            rt = r0 + a * np.cos(m * t)
            xt = rt * np.cos(t)
            yt = rt * np.sin(t)
            if angle != 0:
                rad = np.radians(angle)
                cos_a, sin_a = np.cos(rad), np.sin(rad)
                xt_rot = xt * cos_a - yt * sin_a
                yt_rot = xt * sin_a + yt * cos_a
                xt, yt = xt_rot, yt_rot
            self.boundary_pts = np.vstack([self.center[0] + xt, self.center[1] + yt]).T

        def inside(self, x):
            dx = x[:, 0] - self.center[0]
            dy = x[:, 1] - self.center[1]
            if self.angle != 0:
                rad = np.radians(self.angle)
                cos_a, sin_a = np.cos(rad), np.sin(rad)
                dx, dy = dx * cos_a + dy * sin_a, -dx * sin_a + dy * cos_a
            r = np.sqrt(dx**2 + dy**2)
            t = np.arctan2(dy, dx)
            r_boundary = self.r0 + self.a * np.cos(self.m * t)
            return r <= r_boundary + 1e-12

        def on_boundary(self, x):
            from scipy.spatial import cKDTree
            tree = cKDTree(self.boundary_pts)
            dist, _ = tree.query(x)
            return dist < 1e-2

        def boundary_normal(self, x):
            dx = x[:, 0] - self.center[0]
            dy = x[:, 1] - self.center[1]
            t = np.arctan2(dy, dx)
            
            # r' = -a*m*sin(m*t)
            r = self.r0 + self.a * np.cos(self.m * t)
            dr = -self.a * self.m * np.sin(self.m * t)
            
            # Normal is gradient of F(r, t) = r - (r0 + a*cos(m*t))
            # grad F in polar is (e_r, e_t/r) * (dF/dr, dF/dt)
            # dF/dr = 1, dF/dt = a*m*sin(m*t)
            # e_r = (cos t, sin t), e_t = (-sin t, cos t)
            
            cos_t = np.cos(t)
            sin_t = np.sin(t)
            df_dr = 1.0
            df_dt = self.a * self.m * np.sin(self.m * t)
            
            df_dx = df_dr * cos_t - (df_dt / (r+1e-12)) * sin_t
            df_dy = df_dr * sin_t + (df_dt / (r+1e-12)) * cos_t
            
            grad = np.vstack([df_dx, df_dy]).T
            norm = np.linalg.norm(grad, axis=1, keepdims=True)
            return grad / (norm + 1e-12)

        def random_boundary_points(self, n):
            idx = np.random.choice(self.boundary_pts.shape[0], n, replace=False)
            return self.boundary_pts[idx]

        @property
        def area(self):
            return np.pi * self.r0**2 + 0.5 * np.pi * self.a**2

        @property
        def perimeter(self):
            diff = np.diff(self.boundary_pts, axis=0)
            p = np.sum(np.linalg.norm(diff, axis=1))
            p += np.linalg.norm(self.boundary_pts[-1] - self.boundary_pts[0])
            return p

        @property
        def IQ(self):
            return 4 * np.pi * self.area / self.perimeter**2


    @staticmethod
    def _vertices_rounded_rectangle(xmin, xmax, ymin, ymax, r, n_per_corner=10):
        """CCWr /"""
        w, h = xmax - xmin, ymax - ymin
        r = float(min(r, w / 2.0 - 1e-9, h / 2.0 - 1e-9))
        if r <= 1e-12:
            return np.array(
                [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype=np.float64
            )

        def qarc(cx, cy, t0, t1):
            tt = np.linspace(t0, t1, max(4, int(n_per_corner) + 2))[1:]
            return np.stack([cx + r * np.cos(tt), cy + r * np.sin(tt)], axis=1)

        parts = [
            np.array([[xmin + r, ymin], [xmax - r, ymin]], dtype=np.float64),
            qarc(xmax - r, ymin + r, -np.pi / 2, 0),
            np.array([[xmax, ymin + r], [xmax, ymax - r]], dtype=np.float64),
            qarc(xmax - r, ymax - r, 0, np.pi / 2),
            np.array([[xmax - r, ymax], [xmin + r, ymax]], dtype=np.float64),
            qarc(xmin + r, ymax - r, np.pi / 2, np.pi),
            np.array([[xmin, ymax - r], [xmin, ymin + r]], dtype=np.float64),
            qarc(xmin + r, ymin + r, np.pi, 1.5 * np.pi),
        ]
        v = np.vstack(parts)
        dup = np.concatenate([[True], np.max(np.abs(np.diff(v, axis=0)), axis=1) > 1e-10])
        v = v[dup]
        if len(v) > 1 and np.max(np.abs(v[0] - v[-1])) < 1e-10:
            v = v[:-1]
        return v

    @staticmethod
    def polygon_rounded_rectangle(xmin, xmax, ymin, ymax, r, n_per_corner=10):
        """  Shape.Polygon"""
        v = Shape._vertices_rounded_rectangle(
            xmin, xmax, ymin, ymax, r, n_per_corner
        )
        return Shape.Polygon(v.tolist())

    @staticmethod
    def _vertices_notched_rectangle(
        xmin, xmax, ymin, ymax, notch_half_width, notch_depth, top=True
    ):
        """
         CCW
        notch_half_width: notch_depth>0 
        """
        if top:
            return np.array(
                [
                    [xmin, ymin],
                    [xmax, ymin],
                    [xmax, ymax],
                    [notch_half_width, ymax],
                    [notch_half_width, ymax - notch_depth],
                    [-notch_half_width, ymax - notch_depth],
                    [-notch_half_width, ymax],
                    [xmin, ymax],
                ],
                dtype=np.float64,
            )
        return np.array(
            [
                [xmin, ymax],
                [xmax, ymax],
                [xmax, ymin],
                [notch_half_width, ymin],
                [notch_half_width, ymin + notch_depth],
                [-notch_half_width, ymin + notch_depth],
                [-notch_half_width, ymin],
                [xmin, ymin],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def polygon_notched_rectangle(
        xmin, xmax, ymin, ymax, notch_half_width, notch_depth, top=True
    ):
        """ top/bottom """
        cx = 0.5 * (xmin + xmax)
        v = Shape._vertices_notched_rectangle(
            xmin - cx,
            xmax - cx,
            ymin,
            ymax,
            notch_half_width,
            notch_depth,
            top=top,
        )
        v[:, 0] += cx
        return Shape.Polygon(v.tolist())

    @staticmethod
    def polygon_regular_n_gon(n, circumradius, center=(0.0, 0.0), angle0=0.0):
        """ n  circumradiusangle0 """
        n = int(n)
        c = np.asarray(center, dtype=np.float64)
        ang = angle0 + np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        v = c + circumradius * np.stack([np.cos(ang), np.sin(ang)], axis=1)
        return Shape.Polygon(v.tolist())

    @staticmethod
    def polygon_regular_octagon(circumradius=1.0, center=(0.0, 0.0)):
        """"""
        return Shape.polygon_regular_n_gon(8, circumradius, center, angle0=np.pi / 8.0)

    @staticmethod
    def polygon_regular_star(n_points, r_tip, r_valley, center=(0.0, 0.0), angle0=0.0):
        """
         2n n  r_tip r_valley
        n_points=5 n_points=8 
        """
        n = int(n_points)
        c = np.asarray(center, dtype=np.float64)
        verts = []
        for k in range(2 * n):
            ang = angle0 + (k * np.pi / n)
            rad = r_tip if (k % 2 == 0) else r_valley
            verts.append(c + rad * np.array([np.cos(ang), np.sin(ang)]))
        return Shape.Polygon(np.vstack(verts).tolist())

    @staticmethod
    def _vertices_sawtooth_rectangle(width, height, n_teeth, tooth_height, cx=0.0, cy=0.0):
        """CCW"""
        w, h, th = float(width), float(height), float(tooth_height)
        n_teeth = max(1, int(n_teeth))
        x0, y0 = cx - w / 2.0, cy - h / 2.0
        y_top = y0 + h
        xs = np.linspace(x0 + w, x0, 2 * n_teeth + 1)
        pts = [[x0, y0], [x0 + w, y0], [x0 + w, y_top]]
        for i in range(1, len(xs)):
            yi = y_top + (th if (i % 2 == 1) else 0.0)
            pts.append([xs[i], yi])
        return np.array(pts, dtype=np.float64)

    @staticmethod
    def polygon_sawtooth_rectangle(width, height, n_teeth, tooth_height, center=(0.0, 0.0)):
        """FEKO/COMSOL """
        cx, cy = center
        v = Shape._vertices_sawtooth_rectangle(width, height, n_teeth, tooth_height, cx, cy)
        return Shape.Polygon(v.tolist())

    @staticmethod
    def _vertices_cross(arm_half_length, arm_half_width):
        """CCW"""
        t, L = float(arm_half_width), float(arm_half_length)
        return np.array(
            [
                [-t, -L],
                [t, -L],
                [t, -t],
                [L, -t],
                [L, t],
                [t, t],
                [t, L],
                [-t, L],
                [-t, t],
                [-L, t],
                [-L, -t],
                [-t, -t],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def polygon_cross(arm_half_length, arm_half_width):
        """"""
        return Shape.Polygon(Shape._vertices_cross(arm_half_length, arm_half_width).tolist())

    @staticmethod
    def polygon_arrow(body_half_width=0.30, body_length=0.70,
                      head_half_width=0.70, head_length=0.50,
                      center=(0.0, 0.0)):
        """
          +y
        body 2*body_half_width body_length 
        head 2*head_half_width head_length 
         y  center
         C_local+   
        """
        bw, bl = float(body_half_width), float(body_length)
        hw, hl = float(head_half_width), float(head_length)
        cx, cy = float(center[0]), float(center[1])
        y0 = cy - bl / 2.0
        y1 = cy + bl / 2.0
        y2 = cy + bl / 2.0 + hl
        v = np.array([
            [cx - bw, y0],
            [cx + bw, y0],
            [cx + bw, y1],
            [cx + hw, y1],
            [cx,      y2],
            [cx - hw, y1],
            [cx - bw, y1],
        ], dtype=np.float64)
        return Shape.Polygon(v.tolist())

    @staticmethod
    def polygon_thin_cross(arm_half_length=0.90, arm_half_width=0.10):
        """
        arm_half_width  arm_half_length
          H_global  C_local
         polygon_cross(0.9,0.22) 
        """
        return Shape.polygon_cross(
            arm_half_length=float(arm_half_length),
            arm_half_width=float(arm_half_width)
        )

    @staticmethod
    def polygon_tank_profile(scale=1.5, center=(0.0, 0.0), face_left=True):
        """
        
         C_local
         +  + 
         scale=1.5 50%
        face_left=True False 
        """
        v = np.array(
            [
                [-1.10, -0.46],
                [0.92, -0.46],
                [1.02, -0.38],
                [1.02, -0.16],
                [1.20, -0.16],
                [1.20,  0.16],
                [1.02,  0.16],
                [1.02,  0.38],
                [0.92,  0.46],
                [-1.10, 0.46],
                [-1.22, 0.34],
                [-1.22, -0.34],
            ],
            dtype=np.float64,
        )
        if bool(face_left):
            v[:, 0] = -v[:, 0]
        v = v * float(scale)
        v += np.asarray(center, dtype=np.float64)
        return Shape.Polygon(v.tolist())

    @staticmethod
    def polygon_airplane_profile(scale=1.0, center=(0.0, 0.0), face_right=False):
        """
        
         + 
        
        face_right=Falseface_right=True 
        """
        v = np.array(
            [
                [1.28,  0.00],
                [0.96,  0.06],
                [0.64,  0.07],
                [0.18,  0.10],
                [-0.10, 0.72],
                [-0.34, 0.84],
                [-0.56, 0.26],
                [-0.74, 0.26],
                [-0.92, 0.26],
                [-1.16, 0.98],
                [-1.36, 0.58],
                [-1.02, 0.12],
                [-1.56, 0.06],
                [-1.56, -0.06],
                [-1.02, -0.12],
                [-1.36, -0.58],
                [-1.16, -0.98],
                [-0.92, -0.26],
                [-0.74, -0.26],
                [-0.56, -0.26],
                [-0.34, -0.84],
                [-0.10, -0.72],
                [0.18, -0.10],
                [0.64, -0.07],
                [0.96, -0.06],
            ],
            dtype=np.float64,
        )
        if not bool(face_right):
            v[:, 0] = -v[:, 0]
        v = v * float(scale)
        v += np.asarray(center, dtype=np.float64)
        return Shape.Polygon(v.tolist())

    @staticmethod
    def polygon_6star_sharp(r_tip=1.0, r_valley=0.25, center=(0.0, 0.0)):
        """
        6 
        r_valley/r_tip  H_global 
        """
        return Shape.polygon_regular_star(
            n_points=6,
            r_tip=float(r_tip),
            r_valley=float(r_valley),
            center=center
        )

    @staticmethod
    def polygon_crescent_solid(R=1.0, chord_cut=0.45, n_pts=80, center=(0.0, 0.0)):
        """
         GeometryDifference 
         [0, 2] 
        chord_cut  (0, R)chord_cut 
             H_global   
        """
        cx, cy = float(center[0]), float(center[1])
        R = float(R)
        d = float(chord_cut)
        d = np.clip(d, 1e-3, R - 1e-3)
        half_ang = np.arccos(d / R)
        theta_arc = np.linspace(np.pi - half_ang, np.pi + half_ang, n_pts)
        arc_pts = np.stack([R * np.cos(theta_arc) + cx,
                            R * np.sin(theta_arc) + cy], axis=1)
        y_top = R * np.sin(half_ang)
        chord_start = np.array([[d + cx,  y_top + cy]])
        chord_end   = np.array([[d + cx, -y_top + cy]])
        # arc_pts[0]  (R cos(-half_ang), R sin(-half_ang)) = (-d, y_top)
        # arc_pts[-1]  (-d, -y_top)
        verts = arc_pts
        return Shape.Polygon(verts.tolist())

    @staticmethod
    def polygon_wide_arc_strip(R_outer=1.0, R_inner=0.55, span_deg=260, n_arc=30,
                               n_cap=8, center=(0.0, 0.0)):
        """
        C/ FEKO/COMSOL 
         + + 
        
          C_local  0.2~0.35
               span_deg   H_global  0.3~0.5
        
        Args:
            R_outer   :  1.0
            R_inner   :  = R_outer - R_inner 0.55
            span_deg  :  220~280 260
            n_arc     :  30
            n_cap     :  8
            center    : 
        """
        import numpy as np
        cx, cy = float(center[0]), float(center[1])
        span = np.radians(float(span_deg))
        start_ang = -span / 2.0
        end_ang   =  span / 2.0
        R_cap = (R_outer - R_inner) / 2.0
        R_mid = (R_outer + R_inner) / 2.0

        outer_ang = np.linspace(start_ang, end_ang, n_arc)
        outer_pts = np.column_stack([
            R_outer * np.cos(outer_ang) + cx,
            R_outer * np.sin(outer_ang) + cy,
        ])
        cap1_cx = R_mid * np.cos(end_ang) + cx
        cap1_cy = R_mid * np.sin(end_ang) + cy
        cap1_ang = np.linspace(end_ang, end_ang + np.pi, n_cap + 2)[1:-1]
        cap1_pts = np.column_stack([
            cap1_cx + R_cap * np.cos(cap1_ang),
            cap1_cy + R_cap * np.sin(cap1_ang),
        ])
        inner_ang = np.linspace(end_ang, start_ang, n_arc)
        inner_pts = np.column_stack([
            R_inner * np.cos(inner_ang) + cx,
            R_inner * np.sin(inner_ang) + cy,
        ])
        cap2_cx = R_mid * np.cos(start_ang) + cx
        cap2_cy = R_mid * np.sin(start_ang) + cy
        cap2_ang = np.linspace(start_ang + np.pi, start_ang + 2 * np.pi, n_cap + 2)[1:-1]
        cap2_pts = np.column_stack([
            cap2_cx + R_cap * np.cos(cap2_ang),
            cap2_cy + R_cap * np.sin(cap2_ang),
        ])
        verts = np.vstack([outer_pts, cap1_pts, inner_pts, cap2_pts])
        return Shape.Polygon(verts.tolist())

    @staticmethod
    def polygon_L_shape(long=1.0, short=0.50, thickness=0.30, center=(0.0, 0.0)):
        """
        L 
        long  thickness+ thickness  short
          H_global   
        """
        cx, cy = float(center[0]), float(center[1])
        L, S, T = float(long), float(short), float(thickness)
        x0 = cx - L / 2.0
        y0 = cy - S / 2.0
        v = np.array([
            [x0,         y0],
            [x0 + L,     y0],
            [x0 + L,     y0 + T],
            [x0 + T,     y0 + T],
            [x0 + T,     y0 + S],
            [x0,         y0 + S],
        ], dtype=np.float64)
        return Shape.Polygon(v.tolist())

    @staticmethod
    def polygon_T_shape(stem_half_width=0.18, stem_height=0.75,
                        cap_half_width=0.80, cap_height=0.28,
                        center=(0.0, 0.0)):
        """
        T  + 
         2*cap_half_width  H_global 
          C_local 
        --
        """
        cx, cy = float(center[0]), float(center[1])
        sw, sh = float(stem_half_width), float(stem_height)
        cw, ch = float(cap_half_width), float(cap_height)
        y_bot = cy - sh / 2.0
        y_mid = cy + sh / 2.0
        y_top = y_mid + ch
        v = np.array([
            [cx - sw,  y_bot],
            [cx + sw,  y_bot],
            [cx + sw,  y_mid],
            [cx + cw,  y_mid],
            [cx + cw,  y_top],
            [cx - cw,  y_top],
            [cx - cw,  y_mid],
            [cx - sw,  y_mid],
        ], dtype=np.float64)
        return Shape.Polygon(v.tolist())

    @staticmethod
    def vertices_standard_pentagram():
        """ 10 """
        return [
            [0.000000, -0.618034],
            [-0.138757, -0.190983],
            [-0.587785, -0.190983],
            [-0.224514, 0.072949],
            [-0.363271, 0.500000],
            [0.000000, 0.236068],
            [0.363271, 0.500000],
            [0.224514, 0.072949],
            [0.587785, -0.190983],
            [0.138757, -0.190983],
        ]

    @staticmethod
    def polygon_standard_pentagram(scale=1.0, center=(0.0, 0.0)):
        v = np.asarray(Shape.vertices_standard_pentagram(), dtype=np.float64)
        v *= float(scale)
        v += np.asarray(center, dtype=np.float64)
        return Shape.Polygon(v.tolist())

    @staticmethod
    def hammersley_sample(n, dim):
        """Generate Hammersley sequence samples"""
        try:
            import skopt
            sampler = skopt.sampler.Hammersly(min_skip=-1, max_skip=-1)
            space = [(0.0, 1.0)] * dim
            return np.array(sampler.generate(space, n))
        except:
            # Fallback to pseudorandom
            return np.random.rand(n, dim)


def _arc_length_resample(xy, n_out):
    """ n_out """
    xy = np.asarray(xy, dtype=np.float64)
    if xy.shape[0] < 2:
        return np.repeat(xy[:1], max(n_out, 1), axis=0)
    if np.allclose(xy[0], xy[-1]):
        xy = xy[:-1]
    m = xy.shape[0]
    ring = np.vstack([xy, xy[0:1]])
    seg = np.diff(ring, axis=0)
    dl = np.linalg.norm(seg, axis=1)
    total = float(dl.sum())
    if total < 1e-15:
        return np.repeat(xy[:1], n_out, axis=0)
    uq = np.linspace(0.0, total, n_out, endpoint=False)
    cum = np.concatenate([[0.0], np.cumsum(dl)])
    idx = np.searchsorted(cum, uq, side="right") - 1
    idx = np.clip(idx, 0, m - 1)
    t = (uq - cum[idx]) / (dl[idx] + 1e-30)
    t = np.clip(t, 0.0, 1.0)
    p0 = xy[idx]
    p1 = xy[(idx + 1) % m]
    return (1.0 - t)[:, None] * p0 + t[:, None] * p1


def _densify_polygon_by_edge(vertices, n_total_target):
    """"""
    v = np.asarray(vertices, dtype=np.float64)
    n = v.shape[0]
    v_next = np.roll(v, -1, axis=0)
    e_len = np.linalg.norm(v_next - v, axis=1)
    per = float(e_len.sum())
    if per < 1e-15:
        return v.copy()
    n_total_target = max(n_total_target, n * 2)
    chunks = []
    for i in range(n):
        n_i = max(2, int(np.round(n_total_target * e_len[i] / per)))
        ts = np.linspace(0.0, 1.0, n_i, endpoint=False)
        chunks.append((1.0 - ts)[:, None] * v[i] + ts[:, None] * v_next[i])
    return np.vstack(chunks)


def _rectangle_dense_polyline(geom, points_per_edge=120):
    xmin, xmax, ymin, ymax = geom.xmin, geom.xmax, geom.ymin, geom.ymax
    corners = np.array(
        [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype=np.float64
    )
    pts = []
    for i in range(4):
        p0, p1 = corners[i], corners[(i + 1) % 4]
        ts = np.linspace(0.0, 1.0, points_per_edge, endpoint=False)
        pts.append((1.0 - ts)[:, None] * p0 + ts[:, None] * p1)
    return np.vstack(pts)


def _ordered_boundary_loop(geom, n_loop=512):
    """
     CCW  (n_loop, 2)
    
    """
    if isinstance(geom, (Shape.Polygon, Shape.StarShaped)):
        dense = _densify_polygon_by_edge(geom.vertices, max(480, 8 * geom.nvertices))
        return _arc_length_resample(dense, n_loop)
    if isinstance(geom, Shape.RectangleGeometry):
        dense = _rectangle_dense_polyline(geom, points_per_edge=140)
        return _arc_length_resample(dense, n_loop)
    if isinstance(geom, Shape.Disk):
        th = np.linspace(0.0, 2.0 * np.pi, n_loop, endpoint=False)
        c = geom.center
        r = geom.radius
        ring = np.stack([np.cos(th), np.sin(th)], axis=1)
        return c + r * ring
    if isinstance(geom, Shape.Ellipse):
        th = np.linspace(0.0, 2.0 * np.pi, n_loop, endpoint=False)
        pl = np.stack(
            [geom.semimajor * np.cos(th), geom.semiminor * np.sin(th)], axis=1
        )
        return geom._to_global(pl)
    if isinstance(geom, (Shape.SuperEllipse, Shape.RadialFourier)):
        return _arc_length_resample(np.asarray(geom.boundary_pts, dtype=np.float64), n_loop)
    pts = geom.random_boundary_points(min(4000, max(800, n_loop * 4)))
    centroid = np.mean(pts, axis=0)
    ang = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    pts = pts[np.argsort(ang)]
    return _arc_length_resample(pts, n_loop)

def _beta_curvature_dimensionless(xy, smooth_sigma=4.0):
    """
     beta = (L / (4*pi^2)) * sum(kappa^2 * ds)
     beta -> 1/beta 

    smooth_sigma mollification  
     
    """
    from scipy.ndimage import gaussian_filter1d

    x = xy[:, 0].copy()
    y = xy[:, 1].copy()
    if smooth_sigma > 0 and len(x) >= 8:
        x = gaussian_filter1d(x, sigma=smooth_sigma, mode="wrap")
        y = gaussian_filter1d(y, sigma=smooth_sigma, mode="wrap")
    dx = np.roll(x, -1) - x
    dy = np.roll(y, -1) - y
    ds = np.sqrt(dx * dx + dy * dy + 1e-20)
    d2x = np.roll(dx, -1) - dx
    d2y = np.roll(dy, -1) - dy
    den = np.power(dx * dx + dy * dy + 1e-20, 1.5)
    kappa = np.abs(dx * d2y - dy * d2x) / den
    L = float(ds.sum())
    integral = float(np.sum(kappa * kappa * ds))
    beta = (L / (4.0 * np.pi * np.pi)) * integral
    return beta


_CURVATURE_BETA_CAP = 60.0


def _curvature_energy_from_beta(beta):
    beta = float(max(beta, 1.0))
    if beta <= 1.0 + 1e-14:
        return 0.0
    t = np.log(beta) / np.log(_CURVATURE_BETA_CAP)
    return float(np.clip(np.tanh(t), 0.0, 1.0))


def _effective_area(geom_inner, outer_bbox_geom):
    if isinstance(geom_inner, Shape.GeometryDifference):
        return max(float(geom_inner.geom_a.area - geom_inner.geom_b.area), 1e-30)
    if hasattr(geom_inner, "area"):
        return max(float(geom_inner.area), 1e-30)
    if outer_bbox_geom is not None and hasattr(outer_bbox_geom, "area"):
        return max(float(outer_bbox_geom.area), 1e-30)
    return 1.0


def _random_interior_points(geom_inner, n):
    try:
        return geom_inner.random_points(n, sampler="pseudo")
    except TypeError:
        return geom_inner.random_points(n)


def _normalized_histogram_entropy(eta, num_bins):
    counts, _ = np.histogram(eta, bins=num_bins, range=(0.0, 1.0))
    total_c = int(np.sum(counts))
    if total_c <= 0:
        return 0.0
    p = counts.astype(np.float64) / total_c
    p = p[p > 0]
    h = -np.sum(p * np.log(p + 1e-30))
    h_max = np.log(num_bins)
    return float(np.clip(h / (h_max + 1e-30), 0.0, 1.0))


def _disk_reference_depth_entropy(area, num_bins, n_mc=60000, seed=0):
    """
      =D/=(A/)
     area 
    """
    cache = getattr(_disk_reference_depth_entropy, "_cache", None)
    if cache is None:
        cache = {}
        setattr(_disk_reference_depth_entropy, "_cache", cache)
    key = (round(float(area), 8), int(num_bins), int(n_mc))
    if key in cache:
        return cache[key]

    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    R = np.sqrt(max(float(area), 1e-30) / np.pi)
    theta = rng.uniform(0.0, 2.0 * np.pi, n_mc)
    r = R * np.sqrt(rng.uniform(0.0, 1.0, n_mc))
    pts = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
    tb = np.linspace(0.0, 2.0 * np.pi, 2000, endpoint=False)
    bdry = np.stack([R * np.cos(tb), R * np.sin(tb)], axis=1)
    tree = cKDTree(bdry)
    d, _ = tree.query(pts)
    eta = np.clip(np.asarray(d, dtype=np.float64).ravel() / (R + 1e-20), 0.0, 1.0)
    val = _normalized_histogram_entropy(eta, num_bins)
    cache[key] = val
    return val


def _convex_hull_fill_deficit(geom_inner):
    """
    1  A/A_ 0 > 0
    """
    from scipy.spatial import ConvexHull

    try:
        if isinstance(geom_inner, (Shape.Disk, Shape.Ellipse, Shape.RectangleGeometry)):
            return 0.0
        if isinstance(geom_inner, Shape.GeometryDifference):
            A = float(geom_inner.geom_a.area - geom_inner.geom_b.area)
            if A <= 1e-30:
                return 0.0
            bo = _ordered_boundary_loop(geom_inner.geom_a, 256)
            bi = _ordered_boundary_loop(geom_inner.geom_b, 256)
            pts = np.vstack([bo, bi])
        elif isinstance(geom_inner, (Shape.Polygon, Shape.StarShaped)):
            A = float(geom_inner.area)
            pts = np.asarray(geom_inner.vertices, dtype=np.float64)
        elif isinstance(geom_inner, (Shape.SuperEllipse, Shape.RadialFourier)):
            A = float(geom_inner.area)
            pts = np.asarray(geom_inner.boundary_pts, dtype=np.float64)
        else:
            return 0.0
        if pts.shape[0] < 3:
            return 0.0
        hull = ConvexHull(pts)
        a_hull = float(hull.volume)
        if a_hull < 1e-30:
            return 0.0
        ratio = float(np.clip(A / a_hull, 0.0, 1.0))
        return float(np.clip(1.0 - ratio, 0.0, 1.0))
    except Exception:
        return 0.0


def calculate_complexity(geom_inner, outer_bbox_geom):
    """
     C_local  [0,1]H_global  [0,1]

    C_local   mollification   = (L/4)ds
      tanhln  [0,1) 1C0

    H_global =D/  H_ref  H_emp
    r = max(0,(H_refH_emp)/H_ref)s = 1A/A_H_global = max(r,s)
     A+(1)B 

    Args:
        geom_inner:  (Shape.* )
        outer_bbox_geom:  geom_inner  area   
    """
    from scipy.spatial import cKDTree

    n_loop = 512
    num_bins = 20

    # ---------- C_local ----------
    C_local = 0.0
    try:
        if isinstance(geom_inner, Shape.GeometryDifference):
            beta_o = _beta_curvature_dimensionless(
                _ordered_boundary_loop(geom_inner.geom_a, n_loop)
            )
            beta_i = _beta_curvature_dimensionless(
                _ordered_boundary_loop(geom_inner.geom_b, n_loop)
            )
            beta = max(beta_o, beta_i)
        else:
            beta = _beta_curvature_dimensionless(
                _ordered_boundary_loop(geom_inner, n_loop)
            )
        C_local = _curvature_energy_from_beta(beta)
    except Exception as e:
        print("Warning: :", e)
        C_local = 0.0

    # ---------- H_global ----------
    H_global = 0.0
    try:
        internal_pts = _random_interior_points(geom_inner, 8000)
        if isinstance(geom_inner, Shape.GeometryDifference):
            b1 = _ordered_boundary_loop(geom_inner.geom_a, 900)
            b2 = _ordered_boundary_loop(geom_inner.geom_b, 900)
            bdry_pts = np.vstack([b1, b2])
        else:
            bdry_pts = _ordered_boundary_loop(geom_inner, 1800)

        if len(internal_pts) == 0:
            H_global = 0.0
        else:
            tree = cKDTree(bdry_pts)
            dists, _ = tree.query(internal_pts)
            dists = np.asarray(dists, dtype=np.float64).ravel()
            dists = np.clip(dists, 0.0, None)

            area = _effective_area(geom_inner, outer_bbox_geom)
            rho = np.sqrt(area / np.pi)
            eta = np.clip(dists / (rho + 1e-20), 0.0, 1.0)
            h_emp = _normalized_histogram_entropy(eta, num_bins)
            h_ref = _disk_reference_depth_entropy(area, num_bins)
            rel = float(np.clip((h_ref - h_emp) / (h_ref + 1e-12), 0.0, 1.0))
            hull = _convex_hull_fill_deficit(geom_inner)
            H_global = float(np.clip(max(rel, hull), 0.0, 1.0))
    except Exception as e:
        print("Warning: :", e)
        H_global = 0.0

    return C_local, H_global

