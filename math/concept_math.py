import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from scipy.integrate import odeint

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)

# Define the trajectory analytically
def trajectory_x(t):
    return t

def trajectory_y(t):
    return 0.3 * np.sin(4 * np.pi * t) * (1 - t) + t

# Compute instantaneous velocity v(t) = d/dt [x(t), y(t)]
def velocity_x(t):
    return 1.0  # dx/dt = 1

def velocity_y(t):
    # dy/dt of 0.3 * sin(4πt) * (1-t) + t
    return 0.3 * (4*np.pi * np.cos(4*np.pi*t) * (1-t) - np.sin(4*np.pi*t)) + 1

def instantaneous_velocity(t):
    """Instantaneous velocity v(t) at time t"""
    return np.array([velocity_x(t), velocity_y(t)])

def compute_mean_velocity(z_t, r, t):
    """
    Compute mean velocity u(z_t, r, t) = (1/(t-r)) * integral_r^t v(tau) dtau
    For simplicity, we assume z_t is on the trajectory at time t
    """
    if t <= r or np.abs(t - r) < 1e-10:
        return 0.0
    
    # Compute the integral of velocity from r to t
    # Since our trajectory is 1D in time, we integrate along the trajectory
    num_points = 100
    tau_vals = np.linspace(r, t, num_points)
    
    # Integrate v_x and v_y separately
    integral_vx = 0
    integral_vy = 0
    
    for i in range(len(tau_vals) - 1):
        dt = tau_vals[i+1] - tau_vals[i]
        integral_vx += velocity_x(tau_vals[i]) * dt
        integral_vy += velocity_y(tau_vals[i]) * dt
    
    # Mean velocity magnitude
    mean_v = np.sqrt(integral_vx**2 + integral_vy**2) / (t - r)
    return mean_v

def compute_mean_velocity_rectified(z_t, r, t):
    """Mean velocity for rectified (straighter) flow"""
    # For rectified flow, use the partially straightened trajectory
    def rect_trajectory_y(tau):
        return 0.1 * np.sin(2 * np.pi * tau) * (1 - tau) + tau
    
    def rect_velocity_y(tau):
        # Derivative of rectified trajectory
        return 0.1 * (2*np.pi * np.cos(2*np.pi*tau) * (1-tau) - np.sin(2*np.pi*tau)) + 1
    
    if t <= r or np.abs(t - r) < 1e-10:
        return 0.0
    
    num_points = 100
    tau_vals = np.linspace(r, t, num_points)
    
    integral_vx = 0
    integral_vy = 0
    
    for i in range(len(tau_vals) - 1):
        dt = tau_vals[i+1] - tau_vals[i]
        integral_vx += velocity_x(tau_vals[i]) * dt
        integral_vy += rect_velocity_y(tau_vals[i]) * dt
    
    mean_v = np.sqrt(integral_vx**2 + integral_vy**2) / (t - r)
    return mean_v

# Create figure with three panels
fig = plt.figure(figsize=(20, 6))

# Panel 1: Curvy trajectory with instantaneous velocity field
t_points = np.linspace(0, 1, 50)
x_traj = trajectory_x(t_points)
y_traj = trajectory_y(t_points)

ax1 = fig.add_subplot(131)
ax1.plot(x_traj, y_traj, 'b-', linewidth=2.5)

# Mark start and end points
ax1.scatter([x_traj[0]], [y_traj[0]], color='green', s=150, zorder=5)
ax1.text(x_traj[0]-0.08, y_traj[0], 'z', fontsize=16, weight='bold', ha='right', va='center')

ax1.scatter([x_traj[-1]], [y_traj[-1]], color='red', s=150, zorder=5)
ax1.text(x_traj[-1]+0.08, y_traj[-1], 'x', fontsize=16, weight='bold', ha='left', va='center')

# Show instantaneous velocity vectors
arrow_indices = [3, 8, 13, 18, 23, 28, 33, 38, 43, 47]
for idx in arrow_indices:
    t_val = t_points[idx]
    v = instantaneous_velocity(t_val)
    
    # Normalize and scale for visibility
    norm = np.linalg.norm(v)
    if norm > 0:
        v_scaled = v / norm * 0.08
    
    ax1.arrow(x_traj[idx], y_traj[idx], v_scaled[0], v_scaled[1],
             head_width=0.015, head_length=0.02, fc='red', ec='red', linewidth=1.5)

# Partially rectified path
rectified_y = 0.1 * np.sin(2 * np.pi * t_points) * (1 - t_points) + x_traj
ax1.plot(x_traj, rectified_y, 'g--', linewidth=2.5, alpha=0.7)

ax1.annotate('Rectify', xy=(0.45, 0.3), fontsize=14, color='darkgreen', 
             weight='bold', ha='center')
ax1.annotate('→', xy=(0.55, 0.3), fontsize=20, color='darkgreen', weight='bold')

ax1.set_title('Panel 1: Curvy Flow Path\n(Instantaneous Velocity Vectors)', fontsize=12, weight='bold')
ax1.set_aspect('equal')
ax1.set_xlim(-0.15, 1.25)
ax1.set_ylim(-0.15, 1.25)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)

# Panel 2: Mean Flow training signal surface (mathematically consistent)
ax2 = fig.add_subplot(132, projection='3d')

r_vals = np.linspace(0, 1, 25)
t_vals = np.linspace(0, 1, 25)
R, T = np.meshgrid(r_vals, t_vals)

# Compute actual mean flow surface
mean_flow_surface = np.zeros_like(R)
for i in range(len(r_vals)):
    for j in range(len(t_vals)):
        if T[j, i] > R[j, i] + 0.02:  # Only valid when t > r
            # Compute mean velocity at this (r, t) point
            z_t = np.array([trajectory_x(T[j, i]), trajectory_y(T[j, i])])
            mean_flow_surface[j, i] = compute_mean_velocity(z_t, R[j, i], T[j, i])
        else:
            mean_flow_surface[j, i] = np.nan

# Normalize for visualization
valid_values = mean_flow_surface[~np.isnan(mean_flow_surface)]
if len(valid_values) > 0:
    vmin, vmax = np.percentile(valid_values, [5, 95])
else:
    vmin, vmax = 0, 1

surf = ax2.plot_surface(R, T, mean_flow_surface, cmap='YlOrRd', alpha=0.7, 
                        vmin=vmin, vmax=vmax)

ax2.set_xlabel('r (start time)', fontsize=11)
ax2.set_ylabel('t (end time)', fontsize=11)
ax2.set_zlabel('Mean velocity magnitude', fontsize=11)
ax2.set_title('Panel 2: MeanFlow Training Signal\n(From Curved Flow)', fontsize=12, weight='bold')
ax2.view_init(elev=25, azim=225)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_zticks([])
ax2.xaxis.pane.fill = False
ax2.yaxis.pane.fill = False
ax2.zaxis.pane.fill = False
ax2.xaxis.pane.set_edgecolor('white')
ax2.yaxis.pane.set_edgecolor('white')
ax2.zaxis.pane.set_edgecolor('white')
ax2.grid(False)
ax2.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0)) 


# Panel 3: Mean flow for rectified trajectory
ax3 = fig.add_subplot(133, projection='3d')

mean_flow_rectified = np.zeros_like(R)
for i in range(len(r_vals)):
    for j in range(len(t_vals)):
        if T[j, i] > R[j, i] + 0.02:
            # Use rectified trajectory
            z_t = np.array([trajectory_x(T[j, i]), 
                           0.1 * np.sin(2*np.pi*T[j, i]) * (1 - T[j, i]) + T[j, i]])
            mean_flow_rectified[j, i] = compute_mean_velocity_rectified(z_t, R[j, i], T[j, i])
        else:
            mean_flow_rectified[j, i] = np.nan

surf2 = ax3.plot_surface(R, T, mean_flow_rectified, cmap='Greens', alpha=0.7,
                         vmin=vmin, vmax=vmax)

ax3.set_xlabel('r (start time)', fontsize=11)
ax3.set_ylabel('t (end time)', fontsize=11)
ax3.set_zlabel('Mean velocity magnitude', fontsize=11)
ax3.set_title('Panel 3: Rectified + MeanFlow\n(Smoother Training Signal)', fontsize=12, weight='bold')
ax3.view_init(elev=25, azim=225)
ax3.set_xticks([])
ax3.set_yticks([])
ax3.set_zticks([])
ax3.xaxis.pane.fill = False
ax3.yaxis.pane.fill = False
ax3.zaxis.pane.fill = False
ax3.xaxis.pane.set_edgecolor('white')
ax3.yaxis.pane.set_edgecolor('white')
ax3.zaxis.pane.set_edgecolor('white')
ax3.grid(False)
ax3.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0)) 


# Viewpoint
ax2.view_init(elev=25, azim=135)
ax3.view_init(elev=25, azim=135)


#fig.suptitle('Two-Stage Approach: Rectified Flow + MeanFlow\n(Mathematically Consistent)', 
#             fontsize=16, weight='bold', y=1.02)

plt.tight_layout()
plt.savefig('meanflow_mathematical.png', dpi=300, bbox_inches='tight')
plt.savefig('meanflow_mathematical.eps', format='eps', bbox_inches='tight')
plt.show()

print("Mathematically consistent visualization created!")
print("\nThe mean flow surfaces now represent actual integrals of instantaneous velocity")
print("Panel 2: Mean flow from the highly curved trajectory")
print("Panel 3: Mean flow from the partially rectified (less curved) trajectory")