import numpy as np
from .reference_elements import ReferenceInterval, ReferenceTriangle
np.seterr(invalid='ignore', divide='ignore')

def lagrange_points(cell, degree):
    """Construct the locations of the equispaced Lagrange nodes on cell.

    :param cell: the :class:`~.reference_elements.ReferenceCell`
    :param degree: the degree of polynomials for which to construct nodes.

    :returns: a rank 2 :class:`~numpy.array` whose rows are the
        coordinates of the nodes.

    The implementation of this function is left as an :ref:`exercise
    <ex-lagrange-points>`.

    """
    # First generate the set of points {(i/p, j/p): 0 <= i+j <= p}
    # Then place these points on the reference cell in topological order

    # first, add the vertices of the cell (d=0)
    vertices = cell.vertices
    l_points = np.asarray(vertices, dtype=np.double)

    # then add the edge points  (d=1)
    intervals = dict(sorted(cell.topology[1].items())) # sort by key to ensure consistent ordering
    for edge in intervals.values():
        x0 = vertices[edge[0]]
        x1 = vertices[edge[1]]
        
        # create p - 1 equispaced points on the edge between x0 and x1
        e_points = np.linspace(x0, x1, degree + 1)[1:-1]  # exclude endpoints
        if e_points.size != 0:
            l_points = np.concatenate((l_points, e_points))

    # finally, add the interior points in arbitrary order (d=2)
    # interior points are arranged on a regular grid so they lie on the interpolation of two edge points with same y-coordinate.
    if cell.dim == 2 and degree >= 3:
        for i in range(1, degree - 1):
            x_0 = [0., i] # first interval point has x-coord. 0 and y-coord i.
            
            num_interval_points = degree - i + 1
            x_end = [num_interval_points - 1, i] # before last point on the row

            interior_points = 1 / degree * np.linspace(x_0, x_end, num_interval_points)[1:-1] 
            l_points = np.concatenate((l_points, interior_points))

    return l_points

def vandermonde_matrix(cell, degree, points, grad=False):
    """Construct the generalised Vandermonde matrix for polynomials of the
    specified degree on the cell provided.

    :param cell: the :class:`~.reference_elements.ReferenceCell`
    :param degree: the degree of polynomials for which to construct the matrix.
    :param points: a list of coordinate tuples corresponding to the points.
    :param grad: whether to evaluate the Vandermonde matrix or its gradient.

    :returns: the generalised :ref:`Vandermonde matrix <sec-vandermonde>`

    The implementation of this function is left as an :ref:`exercise
    <ex-vandermonde>`.
    """
    points = np.asarray(points, dtype=np.double)

    from itertools import product
    # powers for 2D
    # powers = [(i,j) for i in range(degree+1) for j in range(degree+1-i)] 
    # generalised to any dimension using multi-indices
    powers = [alpha for alpha in product(range(degree+1), repeat=cell.dim) if sum(alpha) <= degree]
    powers.sort(key=lambda alpha: (sum(alpha), tuple(reversed(alpha)))) # sort by total degree, then lexicographically but in a reversed way (i.e., decreasing x and increasing y)
    powers = np.array(powers, dtype=int)
    
    num_points = points.shape[0]
    V = np.zeros((num_points, len(powers)))

    for idx, alpha in enumerate(powers):
        # V[:, idx] = points[:, 0] ** i * points[:, 1] ** j # only works for 2D
        V[:, idx] = np.prod(points ** alpha , axis=1)  # generalised to any dimension
    
    # If grad is True, replace each entry by a vector of partial derivatives
    if grad:
        # Gradient of Vandermonde shape (num_points, len(powers), dim)
        V_grad = np.zeros((num_points, len(powers), cell.dim))

        for j, alpha in enumerate(powers):
            # Compute the gradient of the monomial analytically:
            # d/dx_k (x_1^{alpha_1} ... x_d^{alpha_d}) = alpha_k * x_k^{alpha_k-1} * Π_{l≠k} x_l^{α_l}
            for k in range(cell.dim):
                grad_alpha = np.array(alpha, dtype=int) 
                if grad_alpha[k] > 0:
                    grad_alpha[k] -= 1 # decrement the k-th entry power
                    V_grad[:, j, k] = alpha[k] * np.prod(points ** grad_alpha, axis=1)
                else:
                    V_grad[:, j, k] = 0.0 # derivative is zero if power is zero
        return V_grad

    return V

class FiniteElement(object):
    def __init__(self, cell, degree, nodes, entity_nodes=None):
        """A finite element defined over cell.

        :param cell: the :class:`~.reference_elements.ReferenceCell`
            over which the element is defined.
        :param degree: the
            polynomial degree of the element. We assume the element
            spans the complete polynomial space.
        :param nodes: a list of coordinate tuples corresponding to
            point evaluation node locations on the element.
        :param entity_nodes: a dictionary of dictionaries such that
            entity_nodes[d][i] is the list of nodes associated with
            entity `(d, i)` of dimension `d` and index `i`.

        Most of the implementation of this class is left as exercises.
        """

        #: The :class:`~.reference_elements.ReferenceCell`
        #: over which the element is defined.
        self.cell = cell
        #: The polynomial degree of the element. We assume the element
        #: spans the complete polynomial space.
        self.degree = degree
        #: The list of coordinate tuples corresponding to the nodes of
        #: the element.
        self.nodes = nodes
        #: A dictionary of dictionaries such that ``entity_nodes[d][i]``
        #: is the list of nodes associated with entity `(d, i)`.
        self.entity_nodes = entity_nodes

        if entity_nodes:
            #: ``nodes_per_entity[d]`` is the number of entities
            #: associated with an entity of dimension d.
            self.nodes_per_entity = np.array([len(entity_nodes[d][0])
                                              for d in range(cell.dim+1)])

        # Replace this exception with some code which sets
        # self.basis_coefs
        # to an array of polynomial coefficients defining the basis functions.

        # By definition: VxC = I  =>  C = V^{-1}
        self.basis_coefs = np.linalg.inv(
            vandermonde_matrix(cell, degree, nodes)
        )

        #: The number of nodes in this element.
        self.node_count = nodes.shape[0]

    def tabulate(self, points, grad=False):
        """Evaluate the basis functions of this finite element at the points
        provided.

        :param points: a list of coordinate tuples at which to
            tabulate the basis.
        :param grad: whether to return the tabulation of the basis or the
            tabulation of the gradient of the basis.

        :result: an array containing the value of each basis function
            at each point. If `grad` is `True`, the gradient vector of
            each basis vector at each point is returned as a rank 3
            array. The shape of the array is (points, nodes) if
            ``grad`` is ``False`` and (points, nodes, dim) if ``grad``
            is ``True``.

        The implementation of this method is left as an :ref:`exercise
        <ex-tabulate>`.

        """
        # Compute the Vandermonde matrix at the points provided
        V = vandermonde_matrix(self.cell, self.degree, points, grad=grad)

        if grad:
            T = np.einsum('ijk,jl->ilk', V, self.basis_coefs)  # use Einstein summation to handle a rank 3 array
        else:
            # Define the tabulation matrix by multiplying V by the basis coefficient matrix C
            T = V @ self.basis_coefs
        return T

    def interpolate(self, fn):
        """Interpolate fn onto this finite element by evaluating it
        at each of the nodes.

        :param fn: A function ``fn(X)`` which takes a coordinate
           vector and returns a scalar value.

        :returns: A vector containing the value of ``fn`` at each node
           of this element.

        The implementation of this method is left as an :ref:`exercise
        <ex-interpolate>`.

        """

        fn_values = np.array([fn(node) for node in self.nodes])

        return fn_values

    def __repr__(self):
        return "%s(%s, %s)" % (self.__class__.__name__,
                               self.cell,
                               self.degree)


class LagrangeElement(FiniteElement):
    def __init__(self, cell, degree):
        """An equispaced Lagrange finite element.

        :param cell: the :class:`~.reference_elements.ReferenceCell`
            over which the element is defined.
        :param degree: the
            polynomial degree of the element. We assume the element
            spans the complete polynomial space.

        The implementation of this class is left as an :ref:`exercise
        <ex-lagrange-element>`.
        """

        nodes = lagrange_points(cell, degree)
        # Use lagrange_points to obtain the set of nodes.  Once you
        # have obtained nodes, the following line will call the
        # __init__ method on the FiniteElement class to set up the
        # basis coefficients.

        # Assuming Lagrange are listed in entity order.
        import math

        """A dictionary of dictionaries that lists the nodes associated with 
        each topological entity in the reference cell."""
        entity_nodes = {}


        node_idx = 0 # tracks the index in the nodes array

        for d in range(cell.dim + 1): # loop over entity dimensions
            num_entity_nodes = math.comb(degree - 1, d) # number of nodes required to be associated with each entity of dim. d
            entity_nodes[d] = {}

            for i in range(cell.entity_counts[d]): # loop over entities of dimension d
                entity_nodes[d][i] = [node_idx + k for k in range(num_entity_nodes)] # assign the lagrange points in order
                node_idx += num_entity_nodes # advance the point nodes       
        
        super(LagrangeElement, self).__init__(cell, degree, nodes, entity_nodes=entity_nodes)
