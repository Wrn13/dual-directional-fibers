"""
Fiber-based fixed point location in the Lorenz system
    f(v)[0] = s*(v[1]-v[0])
    f(v)[1] = r*v[0] - v[1] - v[0]*v[2]
    f(v)[2] = v[0]*v[1] - b*v[2]

Reference:
http://www.emba.uvm.edu/~jxyang/teaching/Math266notes13.pdf
https://en.wikipedia.org/wiki/Lorenz_system
"""

import numpy as np
import matplotlib.pyplot as pt
import scipy.integrate as si
import dfibers.traversal as tv
import dfibers.numerical_utilities as nu
import dfibers.fixed_points as fx
import dfibers.solvers as sv
from mpl_toolkits.mplot3d import Axes3D

N = 3
s, b, r = 10, 8./3., 28

def f(v):
    return np.array([
        s*(v[1,:]-v[0,:]),
        r*v[0,:] - v[1,:] - v[0,:]*v[2,:],
        v[0,:]*v[1,:] - b*v[2,:],
    ])

def ef(v):
    return 0.001*np.ones((N,1))

def Df(v):
    Dfv = np.empty((v.shape[1],3,3))
    Dfv[:,0,0], Dfv[:,0,1], Dfv[:,0,2] = -s, s, 0
    Dfv[:,1,0], Dfv[:,1,1], Dfv[:,1,2] = r - v[2], -1, -v[0]
    Dfv[:,2,0], Dfv[:,2,1], Dfv[:,2,2] = v[1], v[0], -b
    return Dfv

if __name__ == "__main__":

    # Collect attractor points
    t = np.arange(0,40,0.01)
    v = np.ones((N,1))
    A = si.odeint(lambda v, t: f(v.reshape((N,1))).flatten(), v.flatten(), t).T

    # Set up fiber arguments
    v = np.zeros((N,1))
    # c = np.random.randn(N,1)
    c = np.array([[0.83736021, -1.87848114, 0.43935044]]).T
    fiber_kwargs = {
        "f": f,
        "ef": ef,
        "Df": Df,
        "compute_step_amount": lambda trace: (0.1, 0, False),
        "v": v,
        "c": c,
        "terminate": lambda trace: (np.fabs(trace.x[:N,:]) > 50).any(),
        "max_step_size": 1,
        "max_traverse_steps": 2000,
        "max_solve_iterations": 2**5,
    }

    # Visualize strange attractor
    fig = pt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.plot(*A, color='gray', linestyle='-', alpha=0.5)
    br1 = np.sqrt(b*(r-1))
    U = np.array([[0, 0, 0],[br1,br1,r-1],[-br1,-br1,r-1]]).T
    ax.scatter(*U, color='black')

    Curvature = None
    # Run and visualize fiber components, for each fxpt
    xlims, ylims, zlims = [-20,20], [-30,30], [-20,60]

    ### Run direcitonal fiber starting at the origin
    # start from current fxpt
    fiber_kwargs["v"] = U[:,[0]]
    # ax.text(U[0,fc],U[1,fc],U[2,fc], str(fc))

    # Run in one direction
    solution = sv.fiber_solver(**fiber_kwargs)
    V1 = np.concatenate(solution["Fiber trace"].points, axis=1)[:N,:]
    z = solution["Fiber trace"].z_initial

    # Run in other direction (negate initial tangent)
    fiber_kwargs["z"] = -z
    solution = sv.fiber_solver(**fiber_kwargs)
    V2 = np.concatenate(solution["Fiber trace"].points, axis=1)[:N,:]

    # Join fiber segments, restrict to figure limits
    V = np.concatenate((np.fliplr(V1)[:,:-1], V2), axis=1)

    normal, kappa, t = nu.find_curvature_normal(V.T)        # V is (3, M); transpose to (M, 3)
    # Mark the top-k high-curvature samples on the 3D plot
    top_k = 5
    top_idx = np.argpartition(kappa, -top_k)[-top_k:]
    
    highest_curvature_idx = top_idx[-1]
    highest_curvature_point = V[:,highest_curvature_idx]

    ax.scatter(*V[:, highest_curvature_idx], color='red', s=40, zorder=5)
    # Stash for the per-fiber plot below
    if Curvature is None:
        Curvature = {}
    Curvature[0] = (V.copy(), kappa, t)

    V = V[:,::50]
    for i, (lo, hi) in enumerate([xlims, ylims, zlims]):
        V = V[:,(lo < V[i,:]) & (V[i,:] < hi)]
    C = f(V)
    
    ### Perturb highest curvature point by some amount normal to the fiber.

    # Strength list
    perturbation_strengths = np.linspace(-20, 20, 5)

    # Direction 
    normal_direction = normal[highest_curvature_idx]

    # Colors
    jet_5 = pt.colormaps['jet'](np.linspace(0,1,len(perturbation_strengths)))

    for perturbation_strength,color in zip(perturbation_strengths, jet_5):
        displacement = perturbation_strength * normal_direction
        new_initial_point = displacement + highest_curvature_point

        fiber_kwargs["v"] = new_initial_point[:,None]

        # Run in one direction
        solution = sv.fiber_solver(**fiber_kwargs)
        V1p = np.concatenate(solution["Fiber trace"].points, axis=1)[:N,:]
        zp = solution["Fiber trace"].z_initial

        # Run in other direction (negate initial tangent)
        fiber_kwargs["z"] = -zp
        solution = sv.fiber_solver(**fiber_kwargs)
        V2p = np.concatenate(solution["Fiber trace"].points, axis=1)[:N,:]

        # Join fiber segments, restrict to figure limits
        Vp = np.concatenate((np.fliplr(V1p)[:,:-1], V2p), axis=1)

        ax.scatter(*new_initial_point, color="blue")
        ax.plot(*Vp, color=color, linestyle='-', label=f"Strength = {np.round(perturbation_strength,2)}")

    
    # Visualize original fiber fiber
    ax.plot(*V, color='black', linestyle='-')
    ax.quiver(*np.concatenate((V,.1*C),axis=0),color='black')
    ax.quiver(*np.concatenate((highest_curvature_point, normal_direction), axis=0), color='red')

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=15,azim=145)
    ax.legend()
    pt.tight_layout()
    pt.show()

fig2, axes = pt.subplots(len(Curvature), 1, figsize=(7, 3*len(Curvature)), sharex=False)
if len(Curvature) == 1:
    axes = [axes]
for ax2, (fc, (V_fc, kappa_fc, t_fc)) in zip(axes, Curvature.items()):
    ax2.plot(kappa_fc, 'k-')
    top_k = 5
    top_idx = np.argpartition(kappa_fc, -top_k)[-top_k:]
    ax2.scatter(top_idx, kappa_fc[top_idx], color='red', zorder=5)
    ax2.set_title(f"Curvature along fiber through fixed point {fc}")
    ax2.set_xlabel("Sample index along fiber")
    ax2.set_ylabel(r"$\kappa$")
pt.tight_layout()
pt.show()
