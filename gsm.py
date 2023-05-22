import numpy as np

def get_square_sail_vectors(
        boom_half_length: float, 
        billow: float, 
        tip_displacement: float,  
        n_panels: int, 
        full: bool=False
    ):
    billow = max(billow, 1e-9)

    hc = boom_half_length / 2**0.5
    dc = np.sqrt(2)*boom_half_length
    beam_rot = np.arctan(tip_displacement/boom_half_length)

    if billow != 0:
        Rc = billow/2 + boom_half_length**2/4/billow
        zco = billow - Rc
        theta0 = np.pi/2 - np.arccos(np.abs(dc/2)/Rc)

        theta = np.linspace(np.pi - theta0, np.pi + theta0, n_panels)

        vecs = np.ones([n_panels, 3]) * hc
        vecs[:,1] = Rc*np.sin(theta)
        vecs[:,2] = -(Rc*np.cos(theta) - zco)

    else:
        vecs = np.zeros([2,3])
        vecs[:, 0] = hc
        vecs[0, 1] = -hc
        vecs[1, 1] =  hc

    if beam_rot != 0:
        rot_mat = np.array([
                [np.cos(beam_rot),  0, np.sin(beam_rot)],
                [0,                 1, 0          ],
                [-np.sin(beam_rot), 0, np.cos(beam_rot)],
                ])

        for i in range(vecs.shape[0]):
            vecs[i,:] = np.dot(rot_mat, vecs[i,:])

    n_panels = vecs.shape[0]

    if full:
        all_vecs = np.empty([4*(n_panels-1) + 1,3])
        rotated = []
        for i in range(4):
            ang = np.pi / 4 + np.pi/2 * i
            rot_mat = np.array([
                [np.cos(ang), np.sin(ang), 0],
                [-np.sin(ang), np.cos(ang), 0],
                [0, 0, -1]
            ])

            rot = np.matmul(vecs[:-1], rot_mat.T)
            all_vecs[i*(n_panels-1):(i+1)*(n_panels-1)] = rot
            rotated.append(rot)

        all_vecs[-1,:] = all_vecs[0,:]
        return all_vecs

    else:
        return vecs

def get_geometrical_Js_from_square_sail_vectors(vecs: np.ndarray):

    norm  = np.cross(vecs[:-1,:], vecs[1:,:])

    areas = np.linalg.norm(norm, axis=1)/2 # dA == Eq. 4.30 Rios-Reyes

    norm = norm / (areas.reshape([-1,1])*2) # Normalize

    J1 = np.sum(norm * areas[:, np.newaxis], axis=0)

    J2 = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            J2[i, j] = np.sum(norm[:, i] * norm[:, j] * areas)

    J3 = np.zeros((3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                J3[i, j, k] = np.sum(norm[:, i] * norm[:, j] * norm[:, k] * areas)
    
    return J1, J2, J3

def apply_coefficients(
    J1: np.ndarray, J2: np.ndarray, J3: np.ndarray, 
    rho: float, 
    s: float,
    Bf: float,
    epsf: float,
    Bb: float,
    epsb: float,
):
    a2 = Bf*(1-s)*rho + (1-rho)*(epsf*Bf - epsb*Bb)/(epsf+epsb)
    a3 = 1 - rho*s

    return (
        J1 * a3,
        J2 * a2,
        J3 * rho * s
    )

def get_force(
    J1: np.ndarray, J2: np.ndarray, J3: np.ndarray, 
    sunlight_direction: np.ndarray
    ):

    term1 = J1.dot(sunlight_direction) * sunlight_direction
    term2 = J2.dot(sunlight_direction)
    term3 = np.zeros([3])
    for i in range(3):
        term3[i] = 2 * sunlight_direction.dot(J3[:,i,:]).dot(sunlight_direction)

    return - term1 + term2 - term3