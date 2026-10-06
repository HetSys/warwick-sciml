import numpy as np
import matplotlib.pyplot as plt

def plot_mesh(XYZ, c='k', ls='--'):
    wrap = list(range(len(XYZ))) + [0]
    plt.plot(XYZ[wrap, 0], XYZ[wrap, 1], 's'+c+ls)
    plt.xlabel("$x_1$")
    # Set y axis label.
    plt.ylabel("$x_2$")

def plane_stress(E,nu):
    C = np.zeros((3,3))
    De = np.array([[1,nu,0],
                   [nu,1,0], [0,0,(1-nu)/2]])
    De1 = E/(1-nu**2)
    C = np.dot(De1,De)
    return C

def dispstrain_B(xyze, xi, eta):
    natcoord = np.array([[-1, 1, 1, -1],[-1, -1, 1, 1]]) #natural nodal coordinates of a quad element

    # derivatives of shape functions w.r.t. natural coordinates
    dNdnat = np.zeros((2,4))
    dNdnat[0,:]=(1/4)*natcoord[0,:]*(1+natcoord[1,:]*eta)
    dNdnat[1,:]=(1/4)*natcoord[1,:]*(1+natcoord[0,:]*xi)

    # element Jacobian matrix
    Jmat = np.dot(dNdnat,xyze)                      # [2x2 matrix]
    J = np.linalg.det(Jmat)                         # determinant of the Jacobian

    JmatInv = np.linalg.inv(Jmat)                   # inverse of the Jacobian matrix
    dNdx = np.dot(JmatInv, dNdnat)                  # effectively: Jmat^-1 * dNdna

    #displacement-strain matrix
    #linear QUAD element
    dsB = np.zeros((3,8))                            # [3 strain components X 8 DOFs]
    dsB[0, 0::2] = dNdx[0, :]
    dsB[1, 1::2] = dNdx[1, :]
    dsB[2, 0::2] = dNdx[1, :]
    dsB[2, 1::2] = dNdx[0, :]

    return dsB, J

def keval(xyze, Ce, th):
    ke=np.zeros((8,8))                                # create element stiffness matrix (ndof x ndof)

    a = 1/(np.sqrt(3))                                # location of Gauss points (in natural coordinates)
    w = 1                                             # weights
    Gauss = np.array([[-a, a, a, -a],[-a, -a, a, a]]) # Gauss points matrix

    for i in range(4):                                # introduce natural coordinates
        xi=Gauss[0,i]                                 # natural coordinate - horizontal
        eta=Gauss[1,i]                                # natural coordinate - vertical

        dsB, J = dispstrain_B(xyze,xi,eta)           # evaluate dsB matrix and Jacobian

        dsBT=dsB.transpose()
        dot1=np.dot(dsBT,Ce)
        dot2=np.dot(dot1,dsB)
        ke=ke+dot2*J*th*w                             # evaluate element stiffness matrix

    return ke

def sigma_vector(XYZ, C, de):
    a=1/np.sqrt(3)                                       # location of Gauss points (in parent coordinates)
    w=1                                                  # weights
    Gauss = np.array([[-a, a, a, -a],[-a, -a, a, a]])    # Gauss points matrix
    sigma = np.zeros(3)                                  # stress (Voigt notation)

    for j in range(4):                                   # loop over element integration points
        xi=Gauss[0,j]                                    # natural coordinate - horizontal
        eta=Gauss[1,j]                                   # natural coordinate - vertical
        dsB, J = dispstrain_B(XYZ,xi,eta)                # evaluate dsB matrix and Jacobian
        sG1 = np.dot(dsB,de)                             # strain vector (Voigt notation)
        sG2 = sG1
        sigmaGauss = np.dot(C, sG2)                      # stresses at j-th Gauss point within parent element
        sigma = sigma + sigmaGauss*J*w                   # sum of the stresses over all Gauss points within physical element

    return sigma

def quad_solver(x, plot=False):
    """
    Wrapper around simple linear FE solver for trapezoidal geometry from PX912

    Arguments:
        x - vector containing [E (Pa), nu (dimensionless), t (m), fext_top (N)]
        plot - if True, plot mesh before and after applying load

    Returns:
        stress - array of length 3 containing [sigma_xx, sigma_yy, sigma_xy] in units of Pa
    """
    E, nu, th, fext_top = x # unpack inputs

    XYZ = np.array([[0,1],[0,0],[2,0.5],[2,1]])    # nodal coordinates

    ndof = XYZ.size
    BC = np.ones(ndof, dtype=int)

    BC[[0, 1, 2, 3]] = 0
    BCid = BC.nonzero()[0]

    # Applied loads - define the RHS load vector
    rhs = np.zeros(ndof)
    rhs[[1, 7]] = fext_top
    rhs = rhs[BCid]

    C = plane_stress(E,nu)
    ke = keval(XYZ, C, th)
    ke = ke[np.ix_(BCid,BCid)]
    ue = np.linalg.solve(ke, rhs)
    de = np.zeros(XYZ.size)
    de[BCid] = ue
    if plot:
        sdef=10000                                     # scale deformation
        plot_mesh(XYZ)
        plot_mesh(XYZ + sdef*de.reshape(XYZ.shape), c='r', ls='-')
    sigma = sigma_vector(XYZ, C, de)
    return sigma
