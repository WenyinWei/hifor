import numpy as np
from pyna.diff.fieldline import RZ_partial_derivative_of_map_4_Flow_Phi_as_t

def Newton_discrete(BR, BZ, BPhi, R, Z, Phi, x0_RZPhi, h=1.0, epsilon=1e-5, output_trace=True):
    xRZ_trace = [np.asarray(x0_RZPhi[:-1])]
    while len(xRZ_trace)==1 or np.linalg.norm( xRZ_trace[-2] - xRZ_trace[-1] ) > epsilon:
        RZdiff_sols = RZ_partial_derivative_of_map_4_Flow_Phi_as_t(BR, BZ, BPhi, R, Z, Phi, [x0_RZPhi[-1], x0_RZPhi[-1] + 2*np.pi], xRZ_trace[-1], highest_order=1)
        x_mapped = RZdiff_sols[0].sol(x0_RZPhi[-1] + 2*np.pi) 
        Jac_comp = RZdiff_sols[1].sol(x0_RZPhi[-1] + 2*np.pi)
        Jac = np.array( [
            [Jac_comp[2]-1, Jac_comp[0]], 
            [Jac_comp[3], Jac_comp[1]-1]])
        xRZ_trace.append( np.ravel(
            xRZ_trace[-1] - h * np.matmul( np.linalg.inv(Jac), np.array([[x_mapped[0]- xRZ_trace[-1][0] ], [x_mapped[1]- xRZ_trace[-1][1] ]])).T) )
    if output_trace:
        return xRZ_trace
    else: # simply output the element closest to the fixed point
        return xRZ_trace[-1]

