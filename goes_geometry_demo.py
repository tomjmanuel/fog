#!/usr/bin/env python3
"""
GOES Fixed-Grid Geometry Visualizer (3D) — Ray/Intersection FIXED

- Earth-centered coordinate frame:
    +X → 0° lon (Greenwich) in equatorial plane
    +Y → North Pole
    +Z → satellite sub-point longitude on equator
- Satellite is placed at (0, 0, H_center) with
    H_center = perspective_point_height + semi_major_axis
- A robust ray–ellipsoid intersection is used so the viewing ray
  actually hits the ellipsoid surface.

Run:
    python goes_geometry_demo_fixed.py
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ----------------------- Parameters -----------------------
ABI_PROJECTION = {
    "longitude_of_projection_origin": -137.0,   # GOES-18 nominal sub-satellite longitude (deg)
    "semi_major_axis": 6378137.0,               # WGS-84 a (m)
    "semi_minor_axis": 6356752.314245,          # WGS-84 b (m)
    "perspective_point_height": 42164000.0,     # distance from Earth's center (m)
}

LON0_DEG = ABI_PROJECTION["longitude_of_projection_origin"]
r_eq = ABI_PROJECTION["semi_major_axis"]
r_pol = ABI_PROJECTION["semi_minor_axis"]
H_center = ABI_PROJECTION["perspective_point_height"] + r_eq  # satellite distance from Earth's center

# Sample scan angles (radians). Positive x east, positive y north.
x_scan = np.deg2rad(1.2)
y_scan = np.deg2rad(4.0)

# ------------------- Ray/Ellipsoid Intersection -------------------
def ray_intersect_ellipsoid_from_sat(x: float, y: float):
    """
    Cast a ray from satellite S=(0,0,H_center) toward Earth defined by scan angles (x, y).
    Ellipsoid: (X^2 + Y^2)/r_eq^2 + Z^2/r_pol^2 = 1
    Return (sx, sy, sz) on the ellipsoid surface.
    """
    Sx, Sy, Sz = 0.0, 0.0, H_center

    # Direction toward Earth (roughly toward origin): negative Z component.
    ux = -np.cos(y) * np.sin(x)
    uy =  np.sin(y)
    uz = -np.cos(y) * np.cos(x)

    # Quadratic intersection
    A = (ux*ux + uy*uy) / r_eq**2 + (uz*uz) / r_pol**2
    B = 2 * ((Sx*ux + Sy*uy) / r_eq**2 + (Sz*uz) / r_pol**2)
    C = (Sx*Sx + Sy*Sy) / r_eq**2 + (Sz*Sz) / r_pol**2 - 1.0

    disc = B*B - 4*A*C
    if disc < 0:
        raise RuntimeError("Ray misses the ellipsoid (discriminant < 0).")
    t0 = (-B - np.sqrt(disc)) / (2*A)  # near intersection
    t1 = (-B + np.sqrt(disc)) / (2*A)  # far intersection (behind satellite)

    t_candidates = [t for t in (t0, t1) if t > 0]
    if not t_candidates:
        raise RuntimeError("No positive intersection along the ray.")
    t = min(t_candidates)

    sx = Sx + t * ux
    sy = Sy + t * uy
    sz = Sz + t * uz
    return sx, sy, sz

def surface_point_to_lonlat(sx, sy, sz):
    lon0 = np.deg2rad(LON0_DEG)
    lon = lon0 + np.arctan2(sx, sz)
    lat = np.arctan((r_eq**2 / r_pol**2) * (sy / np.sqrt(sx**2 + sz**2)))
    return np.rad2deg(lon), np.rad2deg(lat)

sx, sy, sz = ray_intersect_ellipsoid_from_sat(x_scan, y_scan)
lon_deg, lat_deg = surface_point_to_lonlat(sx, sy, sz)

# Residual check
residual = (sx*sx + sy*sy) / r_eq**2 + (sz*sz) / r_pol**2 - 1.0
print("Intersection (km):", (sx/1e3, sy/1e3, sz/1e3))
print("Lon/Lat (deg): (%.5f, %.5f)" % (lon_deg, lat_deg))
print("Ellipsoid residual (~0):", residual)

# ------------------- Build ellipsoid for plot -------------------
lat = np.linspace(-np.pi/2, np.pi/2, 60)
lon_rel = np.linspace(-np.pi, np.pi, 120)
LAT, LONR = np.meshgrid(lat, lon_rel)
X = r_eq * np.cos(LAT) * np.sin(LONR)
Y = r_pol * np.sin(LAT)
Z = r_eq * np.cos(LAT) * np.cos(LONR)

sat = np.array([0.0, 0.0, H_center])
ray_t = np.linspace(0.0, 1.05, 60)
ray = sat.reshape(3,1) + (np.vstack([sx, sy, sz]).reshape(3,1) - sat.reshape(3,1)) * ray_t

# -------------------------- Plot --------------------------
fig = plt.figure(figsize=(9, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(X/1e6, Y/1e6, Z/1e6, rstride=4, cstride=6, linewidth=0.5, alpha=0.6)
ax.scatter([sat[0]/1e6], [sat[1]/1e6], [sat[2]/1e6], s=40)
ax.scatter([sx/1e6], [sy/1e6], [sz/1e6], s=30)
ax.plot(ray[0]/1e6, ray[1]/1e6, ray[2]/1e6, linewidth=1.5)

axis_len = 10  # in Mm
ax.quiver(0,0,0, axis_len, 0, 0, length=1.0, normalize=False)
ax.quiver(0,0,0, 0, axis_len, 0, length=1.0, normalize=False)
ax.quiver(0,0,0, 0, 0, axis_len, length=1.0, normalize=False)

ax.set_xlabel('+X (toward 0° lon) [Mm]')
ax.set_ylabel('+Y (toward North Pole) [Mm]')
ax.set_zlabel('+Z (toward satellite sub-point) [Mm]')
ax.set_title('GOES Fixed-Grid Geometry (Robust Ray–Ellipsoid Intersection)')

ax.view_init(elev=20, azim=-60)

max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()/1e6
mid_x = (X.max()+X.min())/2/1e6
mid_y = (Y.max()+Y.min())/2/1e6
mid_z = (Z.max()+Z.min())/2/1e6
ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)

plt.tight_layout()
plt.show()
