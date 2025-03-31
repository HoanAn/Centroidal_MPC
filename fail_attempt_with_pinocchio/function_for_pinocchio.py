import numpy as np
def euler_to_quaternion1(roll, pitch, yaw):
    """
    convert euler angle in quaternion
    """

    c_roll = np.cos(roll / 2)
    s_roll = np.sin(roll / 2)
    c_pitch = np.cos(pitch / 2)
    s_pitch = np.sin(pitch / 2)
    c_yaw = np.cos(yaw / 2)
    s_yaw = np.sin(yaw / 2)


    x = s_roll * c_pitch * c_yaw - c_roll * s_pitch * s_yaw
    y = c_roll * s_pitch * c_yaw + s_roll * c_pitch * s_yaw
    z = c_roll * c_pitch * s_yaw - s_roll * s_pitch * c_yaw
    w = c_roll * c_pitch * c_yaw + s_roll * s_pitch * s_yaw

    
    return np.array([x, y, z, w])




def convert_q_Dart_into_q_Pinocchio(q_dart,q_pinocchio):
    "this function try to convert the q value in Dart in q value in Pinocchio"
    "takes as input :"
    "q_dart which are the value in Dart  usually are computed with self.hrp4.GetPositions() or in the setting phase "
    "q_pinocchi0 are just q= np.zeros(model1.nq)  vector of zero with the same dimension of the pinocchio variable "
    " Dart has 30 variable ,   pinocchio 31 , cause use the quaternion for the floating base , while dart use XYZ  euler angle"
    "also the position and orientation positions are exchange"
    

    assert len(q_dart)==len(q_pinocchio)-1 , "ERROR"
    q_pinocchio[0:3]=q_dart[3:6]
    quater=euler_to_quaternion1(q_dart[0], q_dart[1], q_dart[2])
    q_pinocchio[3:7]=quater
    q_pinocchio[7:]=q_dart[6:]

    return q_pinocchio




def permutation_matrix(n):
    "if multiply this matrix for a vector then we swtiched position of first 3 element with the element number 4 5 6 "
    "Note :   P=P'=P^-1   (very useful propriety :) "
    if n < 6:
        raise ValueError("must have 6 element or more")
    
    P = np.eye(n)  # Inizializziamo con la matrice identità
    
    # Scambio dei primi 3 elementi con i successivi 3
    for i in range(3):
        P[i, i], P[i, i+3] = 0, 1
        P[i+3, i+3], P[i+3, i] = 0, 1
    
    return P

def velocity_pin_dart(v_d,v_p):
  assert len(v_p) == len(v_d), 'ERROR, velocity must be same dimension'
  n=len(v_p)
  P=permutation_matrix(n)
  return P@v_p



def convert_matrix(M):
    "the inertia matrix M and corilis matrix C of pinocchio are related with formula M~= PMP  so this function transform "
    
    n=len(M)
    P=permutation_matrix(n)
    return P@M@P




def euler_to_quaternion_matrix(roll, pitch, yaw):
    #not need it for this project
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)



    # x = s_roll * c_pitch * c_yaw - c_roll * s_pitch * s_yaw       derivative of this things 
    # y = c_roll * s_pitch * c_yaw + s_roll * c_pitch * s_yaw
    # z = c_roll * c_pitch * s_yaw - s_roll * s_pitch * c_yaw
    # w = c_roll * c_pitch * c_yaw + s_roll * s_pitch * s_yaw
    
    J = 0.5 * np.array([
      [cr * cp * cy + sr * sp * sy , -sr*sp*cy-cr*cp*sy, -sr*cp*sy-cr*sp*cy  ],
      [-sr*sp*cy+cr*cp*sy,cr*cp*cy-sr*sp*sy, -cr*sp*sy+sr*cp*cy],
      [-sr*cp*sy-cr*sp*cy   ,   -cr*sp*sy-sr*cp*cy, cr*cp*cy+sr*sp*sy],
      [-sr*cp*cy+cr*sr*sy, -cr*sp*cy+sr*cp*sy, -cr*cp*sy+sr*sp*cy]

    ])
    return J

def Jacobian_transformation(q,n):
    T=np.eye(n+1)
    J=euler_to_quaternion_matrix(q[0],q[1],q[2])
    T[3:7,3:6]=J
    return T


