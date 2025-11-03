"""Solve a model Poisson problem with Dirichlet boundary conditions.

If run as a script, the result is plotted. This file can also be
imported as a module and convergence tests run on the solver.
"""
from fe_utils import *
import numpy as np
from numpy import sin, pi
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
from argparse import ArgumentParser


def assemble(fs, f):
    """Assemble the finite element system for the Poisson problem given
    the function space in which to solve and the right hand side
    function."""

    mesh = fs.mesh
    fe = fs.element
    degree = fe.degree
    ref_cell = fe.cell

    # Create a suitable (complete) quadrature rule.

    # The degree of precision is of order degree^2 as the weak form contains
    # products of derivatives of basis functions 
    # (poly. of degree degree - 1)
    quad = gauss_quadrature(ref_cell, degree ** 2)
    quad_points = quad.points
    quad_weights = quad.weights

    # Evaluate the basis functions and their derivatives at the quad. points
    phi = fe.tabulate(quad_points, grad=False) # (points, nodes)
    phi_grad = fe.tabulate(quad_points, grad=True) # (points, nodes, dim)

    # Create the LHS matrix in sparse format
    # Create the RHS vector
    A = sp.lil_matrix((fs.node_count, fs.node_count))
    l = np.zeros(fs.node_count)

    # Step 1: assemble A and l ignoring the Dirichlet BC.
    for c in range(mesh.entity_counts[-1]):
        # global nodes lying on cell c
        c_nodes = fs.cell_nodes[c, :]

        # cell jacobian
        J = mesh.jacobian(c)
        detJ = np.abs(np.linalg.det(J)) 
        invJ = np.linalg.inv(J)

        # RHS same as in Helmholtz pb.
        f_coefs = f.values[c_nodes]
        cell_f = np.einsum('qk,k->q', phi, f_coefs) # contract over nodes k
        cell_f_int = np.einsum('qi,q,q->i', phi, cell_f, quad_weights) # contract over quad q
        l[c_nodes] += cell_f_int * detJ

        # LHS same as in Helmholtz pb. minus the mass term
        invJ_phi_grad = np.einsum('da,qid->qia', invJ, phi_grad) # contract along shared dim d
        invJ_phi_grad_squared = np.einsum('qia,qja->ijq', invJ_phi_grad, invJ_phi_grad) # contract along dim a
        cell_phi_int = np.einsum('ijq,q->ij', invJ_phi_grad_squared, quad_weights) # contract along quad q
        A[np.ix_(c_nodes, c_nodes)] += cell_phi_int * detJ

    # Identify boundary nodes.
    b_nodes = boundary_nodes(fs)

    # Step 2: set boundary rows in l to 0.
    # Step 3+4: set boundary rows in A to 0 and diagonal entry to 1.
    l[b_nodes] = 0
    A[b_nodes, :] = 0
    A[b_nodes, b_nodes] = 1

    return A, l

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


def solve_poisson(degree, resolution, analytic=False, return_error=False):
    """Solve a model Poisson problem on a unit square mesh with
    ``resolution`` elements in each direction, using equispaced
    Lagrange elements of degree ``degree``."""

    # Set up the mesh, finite element and function space required.
    mesh = UnitSquareMesh(resolution, resolution)
    fe = LagrangeElement(mesh.cell, degree)
    fs = FunctionSpace(mesh, fe)

    # Create a function to hold the analytic solution for comparison purposes.
    analytic_answer = Function(fs)
    analytic_answer.interpolate(lambda x: sin(4*pi*x[0])*x[1]**2*(1.-x[1])**2)

    # If the analytic answer has been requested then bail out now.
    if analytic:
        return analytic_answer, 0.0

    # Create the right hand side function and populate it with the
    # correct values.
    f = Function(fs)
    f.interpolate(lambda x: (16*pi**2*(x[1] - 1)**2*x[1]**2 - 2*(x[1] - 1)**2 -
                             8*(x[1] - 1)*x[1] - 2*x[1]**2) * sin(4*pi*x[0]))

    # Assemble the finite element system.
    A, l = assemble(fs, f)

    # Create the function to hold the solution.
    u = Function(fs)

    # Cast the matrix to a sparse format and use a sparse solver for
    # the linear system. This is vastly faster than the dense
    # alternative.
    A = sp.csr_matrix(A)
    u.values[:] = splinalg.spsolve(A, l)

    # Compute the L^2 error in the solution for testing purposes.
    error = errornorm(analytic_answer, u)

    if return_error:
        u.values -= analytic_answer.values

    # Return the solution and the error in the solution.
    return u, error

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
    
def solve_poisson_initial_guess(resolution, degree=1):
    mesh = UnitSquareMesh(resolution, resolution)
    fe = LagrangeElement(mesh.cell, degree)
    fs = FunctionSpace(mesh, fe)

    # Define the same forcing term used in the 4-Laplacian
    f = Function(fs)
    f.interpolate(lambda x: np.exp(3*x[0]*x[1]) *
                  (3*(x[0]**4) + 6*(x[0]**2)*(x[1]**2) + 4*x[0]*x[1] + 3*x[1]))

    # Assemble and solve the Poisson system
    A, l = assemble(fs, f)

    # Apply the same Dirichlet BCs as in the 4-Laplacian
    b_nodes = boundary_nodes(fs)
    b_node_points = mesh.vertex_coords[b_nodes, :]
    b_vals = np.array([b(b_node_point) for b_node_point in b_node_points])

    l[b_nodes] = b_vals
    A[b_nodes, :] = 0
    A[b_nodes, b_nodes] = 1

    u = Function(fs)
    A = sp.csr_matrix(A)
    u.values[:] = splinalg.spsolve(A, l)
    return u

if __name__ == "__main__":

    parser = ArgumentParser(
        description="""Solve a Poisson problem on the unit square.""")
    parser.add_argument("--analytic", action="store_true",
                        help="Plot the analytic solution instead of solving the finite element problem.")
    parser.add_argument("--error", action="store_true",
                        help="Plot the error instead of the solution.")
    parser.add_argument("resolution", type=int, nargs=1,
                        help="The number of cells in each direction on the mesh.")
    parser.add_argument("degree", type=int, nargs=1,
                        help="The degree of the polynomial basis for the function space.")
    args = parser.parse_args()
    resolution = args.resolution[0]
    degree = args.degree[0]
    analytic = args.analytic
    plot_error = args.error

    u, error = solve_poisson(degree, resolution, analytic, plot_error)

    u.plot()
