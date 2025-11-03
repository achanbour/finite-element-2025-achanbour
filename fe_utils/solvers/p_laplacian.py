"""Solve a p-laplacian non-linear PDE.

If run as a script, the result is plotted. This file can also be
imported as a module and convergence tests run on the solver.
"""
from fe_utils import *
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
from argparse import ArgumentParser

def b(x):
    """Define the boundary function"""
    eps = 1e-10
    if x[1] < eps:              # bottom edge
        return 1    
    elif x[1] > 1 - eps:        # top edge
        return np.exp(x[0])     
    elif x[0] < eps:            # left edge
        return 1
    elif x[0] > 1 - eps:
        return np.exp(x[1])     # right edge
    else:
        return 0.0

def assemble_non_linear(fs, f, u, p):
    
    """The steps below are the same as in the linear assembly case"""
    mesh = fs.mesh
    fe = fs.element
    degree = fe.degree if fe.degree == 1 else 1
    ref_cell = fe.cell

    # Create a suitable (complete) quadrature rule.
    # The weak form (the interior residual) is an integral of a product of basis func. derivatives
    # Each basis func. is a poly. of deg. degree (so each derivative is of degree degree - 1)
    quad = gauss_quadrature(ref_cell, degree ** 2)
    quad_points = quad.points
    quad_weights = quad.weights

    # Evaluate the (local) basis func. and their derivatives at quad points 
    phi = fe.tabulate(quad_points, grad=False) #(num_points, num_nodes)
    grad_phi = fe.tabulate(quad_points, grad=True) # (num_points, num_nodes, dim)

    # Create the Jacobian (Gateaux derivative of the residual)
    # Create the residual vector
    J = np.zeros((fs.node_count, fs.node_count))
    R = np.zeros(fs.node_count) 

    """Assemble the linear system"""
    for c in range(mesh.entity_counts[-1]):
        """Residual"""
        # Compute the cell contribution to residual
        c_nodes = fs.cell_nodes[c, :] # (num_nodes, dim)
        jac = mesh.jacobian(c)
        inv_jac = np.linalg.inv(jac)
        det_jac = np.abs(np.linalg.det(jac))

        # Local coefficients
        u_coefs = u.values[c_nodes] # coef of basis func. located on cell c (num_nodes, )
        g_coefs = f.values[c_nodes] # coef of basis func. located on cell c (num_nodes, )

        # Evaluate u and g at quad. points
        g_val = np.einsum('qj,j->q', phi, g_coefs) # contract along nodes index j
        
        # Pull gradients of basis functions to ref. space
        grad_phi_pullback = np.einsum('da,qid->qia', inv_jac, grad_phi) # contract along dim index d

        # Evaluate the gradient of u at quad. points
        grad_u = np.einsum('j, qja-> qa', u_coefs, grad_phi_pullback) # contract along nodes index j
        grad_u_norm2 = np.einsum('qa,qa->q', grad_u, grad_u) # contract along dim index a
        
        grad_u_norm_p_minus_2 = grad_u_norm2 ** ((p-2) / 2) 
        grad_u_norm_p_minus_4 = grad_u_norm2 ** ((p-4) / 2) 

        grad_u_grad_phi = np.einsum('qa,qia->qi', grad_u, grad_phi_pullback) # contract along dim index a

        grad_term = np.einsum('q,qi->qi', grad_u_norm_p_minus_2, grad_u_grad_phi)
        mass_term = np.einsum('q,qi->qi', g_val, phi)

        cell_int = np.einsum('qi,q->i', (grad_term - mass_term), quad_weights) # contract along quad. index q
        R[c_nodes] += cell_int * det_jac

        """Jacobian"""
        # Compute the cell contribution to the Jacobian
        # pre-compute all required terms
        grad_u_dot_grad_phi = np.einsum('qa,qia->qi', grad_u, grad_phi_pullback) # contract along dim. index a
        grad_phi_squared = np.einsum('qja, qia->qji', grad_phi_pullback, grad_phi_pullback) # contract along dim. index a

        term1 = np.einsum('q, qji -> qji', grad_u_norm_p_minus_2, grad_phi_squared)
        grad_u_dot_grad_phi_squared = np.einsum('qi, qj->qij', grad_u_dot_grad_phi, grad_u_dot_grad_phi)
    
        term2 = (p-2) * np.einsum('qij, q->qij', grad_u_dot_grad_phi_squared, grad_u_norm_p_minus_4)
        jac_int = np.einsum('qij,q->ij', (term1 + term2), quad_weights)

        J[np.ix_(c_nodes, c_nodes)] += jac_int * det_jac

    """Enforce Dirichlet BC (strongly)"""
    # Overwrite boundary node rows in the jacobian and the residual
    # boundary residual is nodal diff. (evaluation of u at node point - evaluation of b at node point)
    # boundary jacobian is the evaluation of u_hat at node point

    # Identify boundary nodes 
    b_nodes = boundary_nodes(fs) # list of global node indices

    # Evaluate the boundary function at the boundary node points
    # NOTE: only works when nodes = vertices (e.g., in CG1 finite element)
    b_node_points = mesh.vertex_coords[b_nodes,:] # global coordinates
    b_vals = np.array([b(b_node_point) for b_node_point in b_node_points])

    R[b_nodes] = u.values[b_nodes] - b_vals

    J[b_nodes, :] = 0
    J[:, b_nodes] = 0
    J[b_nodes, b_nodes] = 1

    return R, J

def boundary_nodes(fs):
    """Find the list of boundary nodes in fs. This is a
    unit-square-specific solution. A more elegant solution would employ
    the mesh topology and numbering.
    """
    eps = 1.e-10

    f = Function(fs)

    def on_boundary(x):
        """Return 1 if on the boundary, 0. otherwise."""
        if x[0] < eps or x[0] > 1 - eps or x[1] < eps or x[1] > 1 - eps:
            return 1.
        else:
            return 0.

    f.interpolate(on_boundary)

    return np.flatnonzero(f.values)


def solve_plaplacian(degree, resolution, p, analytic=False, return_error=False):
    """Solve a p-laplacian non-linear PDE on a unit square mesh with
    ``resolution`` elements in each direction, using equispaced
    Lagrange elements of degree 1."""

    # Set up the mesh, finite element and function space required.
    mesh = UnitSquareMesh(resolution, resolution)
    fe = LagrangeElement(mesh.cell, degree)
    fs = FunctionSpace(mesh, fe)

    # Create a function to hold the analytic solution for comparison purposes.
    analytic_answer = Function(fs)
    analytic_answer.interpolate(lambda x: np.exp(x[0]*x[1]))

    # If the analytic answer has been requested then bail out now.
    if analytic:
        return analytic_answer, 0.0

    # Create the right hand side function and populate it with the
    # correct values.
    f = Function(fs)
    f.interpolate(lambda x: np.exp(3*x[0]*x[1])*(3*(x[0]**4) + 6*(x[0]**2)*(x[1]**2)+4*x[0]*x[1]+3*x[1]))

    """Newton iteration"""
    u = Function(fs) # create the function to hold the solution.

    zero = Function(fs) # create a zero function
    zero.values[:] = 0

    from fe_utils.solvers.poisson import solve_poisson_initial_guess
    # Solve the Poisson problem (p=2) and use the solution as initial guess
    u = solve_poisson_initial_guess(resolution, degree)

    tol = 1e-8
    max_iters = 20
    for k in range(max_iters):
        R, J = assemble_non_linear(fs, f, u, p)
        # res_norm = np.linalg.norm(R)
        J = sp.csr_matrix(J) # cast the matrix to sparse format to use a sparse solver
        u_hat_coefs = splinalg.spsolve(J, -R) # compute Newton's update
        
        u_hat = Function(fs) # create a function for Newton's step
        u_hat.values[:] = u_hat_coefs
        step_norm = errornorm(u_hat, zero)  # compute L^2 error of Newton's step
        
        print(f"Iter {k}: error {step_norm:.3e}")
        if step_norm < tol:
            break
        
        u.values[:] += u_hat_coefs

    # Compute the L^2 error in the solution for testing purposes.
    error = errornorm(analytic_answer, u)

    if return_error:
        u.values -= analytic_answer.values

    # Return the solution and the error in the solution.
    return u, error


if __name__ == "__main__":

    parser = ArgumentParser(
        description="""Solv§e a p-Laplacian problem on the unit square.""")
    parser.add_argument("--analytic", action="store_true",
                        help="Plot the analytic solution instead of solving the finite element problem.")
    parser.add_argument("--error", action="store_true",
                        help="Plot the error instead of the solution.")
    parser.add_argument("resolution", type=int, nargs=1,
                        help="The number of cells in each direction on the mesh.")
    parser.add_argument("p", type=int, nargs='?', default=4,
                        help="The Laplacian exponent.")
    parser.add_argument("degree", type=int, nargs='?', default=1,
                        help="The degree of the polynomial basis for the function space.")
    args = parser.parse_args()
    resolution = args.resolution[0]
    degree = args.degree
    p = args.p

    analytic = args.analytic
    plot_error = args.error

    u, error = solve_plaplacian(degree, resolution, p, analytic, plot_error)

    u.plot()