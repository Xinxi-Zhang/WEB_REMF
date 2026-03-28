import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d

# ╔══════════════════════════════════════════════════════╗
# ║  TUNABLE CURVATURE PARAMETERS                       ║
# ║  CURVE_AMP  – amplitude of the sinusoidal wiggle    ║
# ║  CURVE_FREQ – number of full sine cycles in [0,1]   ║
# ╚══════════════════════════════════════════════════════╝
CURVE_AMP  = 0.6   # try 0.5–1.0 for more dramatic curvature
CURVE_FREQ = 2     # number of full sine cycles in [0,1] (original = 2; try 1–6)

# ── Curved trajectory: x(t) = t, y(t) = A*sin(F*2πt)*(1-t) + t ──

def curved_pos(t):
    w = CURVE_FREQ * 2 * np.pi
    return np.array([t, CURVE_AMP * np.sin(w * t) * (1 - t) + t])

def curved_vel(t):
    w = CURVE_FREQ * 2 * np.pi
    vx = 1.0
    vy = CURVE_AMP * (w * np.cos(w * t) * (1 - t) - np.sin(w * t)) + 1
    return np.array([vx, vy])

# ── Rectified (straighter) trajectory: x(t) = t, y(t) = 0.1*sin(2πt)*(1-t) + t ──

def rect_pos(t):
    return np.array([t, 0.1 * np.sin(2 * np.pi * t) * (1 - t) + t])

def rect_vel(t):
    vx = 1.0
    vy = 0.1 * (2*np.pi * np.cos(2*np.pi*t) * (1-t) - np.sin(2*np.pi*t)) + 1
    return np.array([vx, vy])

# ── Metric: angle between v(t) and the straight-line direction pos(t)−pos(r) ──

def angle_diff(vel, direction):
    nv = np.linalg.norm(vel)
    nd = np.linalg.norm(direction)
    if nv < 1e-12 or nd < 1e-12:
        return 0.0
    cos_a = np.clip(np.dot(vel, direction) / (nv * nd), -1.0, 1.0)
    return np.arccos(cos_a)

# ── Build (r, t) grid and compute angle-difference surfaces ──

N = 60
r_vals = np.linspace(0, 1, N)
t_vals = np.linspace(0, 1, N)
R, T = np.meshgrid(r_vals, t_vals)

curved_surf = np.full_like(R, np.nan)
rect_surf   = np.full_like(R, np.nan)

for i in range(N):
    for j in range(N):
        t, r = T[j, i], R[j, i]
        if t > r + 0.02:
            d_curved = curved_pos(t) - curved_pos(r)
            curved_surf[j, i] = np.degrees(angle_diff(curved_vel(t), d_curved))

            d_rect = rect_pos(t) - rect_pos(r)
            rect_surf[j, i] = np.degrees(angle_diff(rect_vel(t), d_rect))

# Shared z-limits so the two surfaces are visually comparable
valid_c = curved_surf[~np.isnan(curved_surf)]
valid_r = rect_surf[~np.isnan(rect_surf)]
z_lo = 0
z_hi = max(np.nanmax(valid_c), np.nanmax(valid_r)) * 1.05

# ── Figure ──

fig = plt.figure(figsize=(20, 6))

# ──── Panel 1: trajectories + velocity arrows ────
ax1 = fig.add_subplot(131)

t_fine = np.linspace(0, 1, 300)
cx = np.array([curved_pos(s)[0] for s in t_fine])
cy = np.array([curved_pos(s)[1] for s in t_fine])
rx = np.array([rect_pos(s)[0] for s in t_fine])
ry = np.array([rect_pos(s)[1] for s in t_fine])

ax1.plot(cx, cy, color='#D8604F', linewidth=6.5, label='Curved flow')
#ax1.plot(rx, ry, 'g-', linewidth=6.5, alpha=0.7, label='Rectified flow')

ax1.scatter([1], [1], color='green', s=150, zorder=5)
#ax1.text(1.08, 1, r'$z$  (noise, $t\!=\!1$)', fontsize=13, weight='bold', ha='left')
ax1.scatter([0], [0], color='red', s=150, zorder=5)
#ax1.text(0.0, 0.08, r'$x$  (clean, $t\!=\!0$)', fontsize=13, weight='bold', ha='center', va='bottom')

# for s in np.linspace(0.06, 0.94, 10):
#     v = -curved_vel(s)
#     p = curved_pos(s)
#     v_sc = v / np.linalg.norm(v) * 0.08
#     ax1.arrow(p[0], p[1], v_sc[0], v_sc[1],
#               head_width=0.015, head_length=0.02,
#               fc='red', ec='red', linewidth=1.5)

# ax1.annotate('Rectify', xy=(0.45, 0.3), fontsize=14,
#              color='darkgreen', weight='bold', ha='center')
# ax1.annotate(r'$\rightarrow$', xy=(0.58, 0.3), fontsize=20,
#              color='darkgreen', weight='bold')

ax1.set_title('Flow Paths\n(with instantaneous velocity)', fontsize=12, weight='bold')
ax1.set_aspect('equal')
ax1.set_xlim(-0.15, 1.25)
ax1.set_ylim(1.25, -0.15)
ax1.set_xticks([]); ax1.set_yticks([])
for sp in ax1.spines.values():
    sp.set_visible(False)

# ──── Panel 2: angle-difference surface – curved flow ────
ax2 = fig.add_subplot(132, projection='3d')
surf2 = ax2.plot_surface(R, T, curved_surf, cmap='Blues', alpha=0.75,
                          vmin=z_lo, vmax=z_hi, edgecolor='none')
ax2.set_xlabel('r', fontsize=11)
ax2.set_ylabel('t',   fontsize=11)
ax2.set_zlabel('Angle diff (°)',  fontsize=11, labelpad=8)
ax2.set_zlim(z_lo, z_hi)
ax2.set_title('Velocity–Direction Angle\n(Curved Flow)', fontsize=12, weight='bold')
ax2.view_init(elev=25, azim=135)
ax2.set_xticks([0, 0.5, 1]); ax2.set_xticklabels(['0', '0.5', '1'], fontsize=8)
ax2.set_yticks([0, 0.5, 1]); ax2.set_yticklabels(['0', '0.5', '1'], fontsize=8)
for axis in (ax2.xaxis, ax2.yaxis):
    axis.pane.fill = False
    axis.pane.set_edgecolor('white')
ax2.zaxis.pane.fill = False
ax2.zaxis.pane.set_edgecolor('white')
ax2.grid(False)

# ──── Panel 3: angle-difference surface – rectified flow ────
ax3 = fig.add_subplot(133, projection='3d')
surf3 = ax3.plot_surface(R, T, rect_surf, cmap='Blues', alpha=0.85,
                          vmin=z_lo, vmax=z_hi * 0.3, edgecolor='none')
ax3.set_xlabel('r', fontsize=11)
ax3.set_ylabel('t',   fontsize=11)
ax3.set_zlabel('Angle diff (°)',  fontsize=11, labelpad=8)
ax3.set_zlim(z_lo, z_hi)
ax3.set_title('Velocity–Direction Angle\n(Rectified Flow)', fontsize=12, weight='bold')
ax3.view_init(elev=25, azim=135)
ax3.set_xticks([0, 0.5, 1]); ax3.set_xticklabels(['0', '0.5', '1'], fontsize=8)
ax3.set_yticks([0, 0.5, 1]); ax3.set_yticklabels(['0', '0.5', '1'], fontsize=8)
for axis in (ax3.xaxis, ax3.yaxis):
    axis.pane.fill = False
    axis.pane.set_edgecolor('white')
ax3.zaxis.pane.fill = False
ax3.zaxis.pane.set_edgecolor('white')
ax3.grid(False)

plt.tight_layout()
plt.savefig('concept_true.png', dpi=300, bbox_inches='tight')
plt.savefig('concept_true.eps', format='eps', bbox_inches='tight')
plt.show()

print("Done.")
print(f"Curved  surface  –  max angle diff: {np.nanmax(valid_c):.1f}°,  mean: {np.nanmean(valid_c):.1f}°")
print(f"Rectified surface – max angle diff: {np.nanmax(valid_r):.1f}°,  mean: {np.nanmean(valid_r):.1f}°")
