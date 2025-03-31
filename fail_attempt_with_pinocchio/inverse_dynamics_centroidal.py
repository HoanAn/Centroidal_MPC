import dartpy as dart
import numpy as np
from utils import *

###############################################################################
#  Calcolo (minimale) della Centroidal Momentum Matrix e di J_hw
###############################################################################
import dartpy as dart
import numpy as np

def skew(v):
    """Restituisce la matrice antisimmetrica di un vettore 3D."""
    return np.array([
        [0.0,    -v[2],  v[1]],
        [v[2],   0.0,   -v[0]],
        [-v[1],  v[0],   0.0]
    ])

def compute_centroidal_momentum_matrix(robot):
    """
    Calcola la Centroidal Momentum Matrix (CMM) A_G, esprimendo i contributi di ciascun link
    correttamente nel world frame e rispetto al centro di massa globale.
    """
    # Numero di gradi di libertà
    n_dofs = robot.getNumDofs()
    # Matrice risultante (6 x N)
    A_G = np.zeros((6, n_dofs))

    # Calcolo del centro di massa globale (nel world frame)
    com_world = robot.getCOM()

    # Iteriamo su tutti i link del robot
    n_links = robot.getNumBodyNodes()
    for i in range(n_links):
        link_i = robot.getBodyNode(i)

        # Posizione del centro di massa del link nel world frame
        link_com_world = link_i.getCOM()

        # Jacobiano 6xN valutato nel CoM del link
        J_link = robot.getJacobian(link_i,        inCoordinatesOf=dart.dynamics.Frame.World())

        # Tensore di inerzia spaziale nel frame locale del link
        I_link_local = link_i.getInertia().getSpatialTensor()

        # Matrice di rotazione per trasformare dal frame del link al world frame
        R_i = link_i.getTransform().rotation()  # Matrice 3x3

        # Costruzione della matrice di trasformazione spaziale (6x6)
        E = np.block([
            [R_i, np.zeros((3, 3))],
            [np.zeros((3, 3)), R_i]
        ])

        # Ruotiamo la matrice d'inerzia nel world frame
        I_link_world = E @ I_link_local @ E.T  # Matrice di inerzia spaziale nel world frame

        # Vettore di traslazione dal CoM del link al CoM globale
        r = link_com_world - com_world

        # Matrice adjoint trasposta per passare dal frame del CoM del link a quello del CoM globale
        X = np.block([
            [np.eye(3), np.zeros((3, 3))],
            [-skew(r), np.eye(3)]
        ])
        X_T = X.T

        # Contributo alla CMM
        A_i = X_T @ I_link_world @ J_link
        A_G += A_i

    return A_G


class InverseDynamics:
    def __init__(self, robot, redundant_dofs, foot_size=0.1, µ=0.5):
        self.robot = robot
        self.dofs = self.robot.getNumDofs()#An: degree of freedom, not number of joints
        self.d = foot_size / 2.
        self.µ = µ

        # define sizes for QP solver
        self.num_contacts = 2
        self.num_contact_dims = self.num_contacts * 6
        self.n_vars = 2 * self.dofs + self.num_contact_dims

        self.n_eq_constraints = self.dofs
        self.n_ineq_constraints = 8 * self.num_contacts

        # initialize QP solver
        self.qp_solver = QPSolver(self.n_vars, self.n_eq_constraints, self.n_ineq_constraints)

        # selection matrix for redundant dofs
        self.joint_selection = np.zeros((self.dofs, self.dofs))
        for i in range(self.dofs):
            joint_name = self.robot.getDof(i).getName()
            if joint_name in redundant_dofs:
                self.joint_selection[i, i] = 1

    def get_joint_torques(self, desired, current, contact):
        #print(f'\t In Inverse Dynamics:')
        #print(f' Actual robot_lfoot_orientation:{current['lfoot']['pos'][0:3]}')
        #print(f' Desired robot_lfoot_orientation:{desired['lfoot']['pos'][0:3]}')
        #print(f' Actual robot_lfoot_position:{current['lfoot']['pos'][3:6]}')
        #print(f' Desired robot_lfoot_position:{desired['lfoot']['pos'][3:6]}\n')
        #print(f' Actual robot_rfoot_orientation:{current['rfoot']['pos'][0:3]}')
        #print(f' Desired robot_rfoot_orientation:{desired['rfoot']['pos'][0:3]}')
        #print(f' Actual robot_rfoot_position:{current['rfoot']['pos'][3:6]}')
        #print(f' Desired robot_rfoot_position:{desired['rfoot']['pos'][3:6]}')

        contact_l = contact == 'lfoot'  or contact == 'ds'
        contact_r = contact == 'rfoot' or contact == 'ds'

        # robot parameters
        lsole = self.robot.getBodyNode('l_sole')
        rsole = self.robot.getBodyNode('r_sole')
        torso = self.robot.getBodyNode('torso')
        base  = self.robot.getBodyNode('body')

        # weights and gains
        tasks = ['lfoot', 'rfoot', 'com', 'torso', 'base', 'joints', 'hw']
        weights   = {'lfoot':  2., 'rfoot':  2., 'com':  2., 'torso': 1., 'base': 1., 'joints': 1.e-2, 'hw':100}
        pos_gains = {'lfoot': 50., 'rfoot': 50., 'com':  5., 'torso': 10., 'base': 10., 'joints': 10.  , 'hw':100}
        vel_gains = {'lfoot': 10., 'rfoot': 10., 'com': 10., 'torso': 2., 'base': 2., 'joints': 1.e-1, 'hw':2}

        # jacobians
        J = {'lfoot' : self.robot.getJacobian(lsole,        inCoordinatesOf=dart.dynamics.Frame.World()),
             'rfoot' : self.robot.getJacobian(rsole,        inCoordinatesOf=dart.dynamics.Frame.World()),
             'com'   : self.robot.getCOMLinearJacobian(     inCoordinatesOf=dart.dynamics.Frame.World()),
             'torso' : self.robot.getAngularJacobian(torso, inCoordinatesOf=dart.dynamics.Frame.World()),
             'base'  : self.robot.getAngularJacobian(base,  inCoordinatesOf=dart.dynamics.Frame.World()),
             'joints': self.joint_selection}

        # jacobians derivatives
        Jdot = {'lfoot' : self.robot.getJacobianClassicDeriv(lsole, inCoordinatesOf=dart.dynamics.Frame.World()),
                'rfoot' : self.robot.getJacobianClassicDeriv(rsole, inCoordinatesOf=dart.dynamics.Frame.World()),
                'com'   : self.robot.getCOMLinearJacobianDeriv(     inCoordinatesOf=dart.dynamics.Frame.World()),
                'torso' : self.robot.getAngularJacobianDeriv(torso, inCoordinatesOf=dart.dynamics.Frame.World()),
                'base'  : self.robot.getAngularJacobianDeriv(base,  inCoordinatesOf=dart.dynamics.Frame.World()),
                'joints': np.zeros((self.dofs, self.dofs))}

        # feedforward terms
        ff = {'lfoot' : desired['lfoot']['acc'],
              'rfoot' : desired['rfoot']['acc'],
              'com'   : desired['com']['acc'],
              'torso' : desired['torso']['acc'],
              'base'  : desired['base']['acc'],
              'joints': desired['joint']['acc'],
              'hw': np.zeros(3)}

        # error vectors
        pos_error = {'lfoot' : pose_difference(desired['lfoot']['pos'] , current['lfoot']['pos'] ),
                     'rfoot' : pose_difference(desired['rfoot']['pos'], current['rfoot']['pos']),
                     'com'   : desired['com']['pos'] - current['com']['pos'],
                     'torso' : rotation_vector_difference(desired['torso']['pos'], current['torso']['pos']),
                     'base'  : rotation_vector_difference(desired['base']['pos'] , current['base']['pos'] ),
                     'joints': desired['joint']['pos'] - current['joint']['pos'],
                     'hw': np.zeros(3)}

        # velocity error vectors
        vel_error = {'lfoot' : desired['lfoot']['vel'] - current['lfoot']['vel'],
                     'rfoot' : desired['rfoot']['vel'] - current['rfoot']['vel'],
                     'com'   : desired['com']['vel']   - current['com']['vel'],
                     'torso' : desired['torso']['vel'] - current['torso']['vel'],
                     'base'  : desired['base']['vel']  - current['base']['vel'],
                     'joints': desired['joint']['vel'] - current['joint']['vel'],
                     'hw': np.zeros(3)}
        

        A_G = compute_centroidal_momentum_matrix(self.robot)
        J_hw = A_G[3:6, :]  # Righe 3..5 -> parte angolare

        # Derivata Jdot_hw (nel contesto attuale, non c'è una funzione nativa DART).
        # Mettiamo semplicemente zero se non l'abbiamo calcolata con un metodo numerico:
        Jdot_hw = np.zeros_like(J_hw)

        # Stampo debug per verificare la formula
        hw_current = J_hw @ current['joint']['vel']
        print("\n[DEBUG] Shape A_G:", A_G.shape, "  Shape J_hw:", J_hw.shape)
        print("[DEBUG] Angular momentum (h_w) attuale calcolato:", hw_current)

        # Se in 'desired' c'è la chiave 'hw', posso calcolare errori
        if 'hw' in desired and 'val' in desired['hw']:
            pos_error['hw'] = desired['hw']['val'] - hw_current
        if 'hw' in desired and 'vel' in desired['hw']:
            vel_error['hw'] = desired['hw']['vel'] - 0.0  # o un valore differente se desiderato


        # Aggiungo j_hw al dictionary dei jacobiani per la fase di costo:
        J['hw'] = J_hw
        Jdot['hw'] = Jdot_hw

        # cost function
        H = np.zeros((self.n_vars, self.n_vars))
        F = np.zeros(self.n_vars)
        q_ddot_indices = np.arange(self.dofs)
        tau_indices = np.arange(self.dofs, 2 * self.dofs)
        f_c_indices = np.arange(2 * self.dofs, self.n_vars)

        for task in tasks:
            H_task =   weights[task] * J[task].T @ J[task]
            F_task = - weights[task] * J[task].T @ (ff[task]
                                                    + vel_gains[task] * vel_error[task]
                                                    + pos_gains[task] * pos_error[task]
                                                    - Jdot[task] @ current['joint']['vel'])

            H[np.ix_(q_ddot_indices, q_ddot_indices)] += H_task
            F[q_ddot_indices] += F_task

        # regularization term for contact forces
        H[np.ix_(f_c_indices, f_c_indices)] += np.eye(len(f_c_indices)) * 1e-6

        # dynamics constraints: M * q_ddot + C - J_c^T * f_c = tau
        inertia_matrix = self.robot.getMassMatrix()
        actuation_matrix = block_diag(np.zeros((6, 6)), np.eye(self.dofs - 6))
        contact_jacobian = np.vstack((contact_l * J['lfoot'], contact_r * J['rfoot']))
        A_eq = np.hstack((inertia_matrix, - actuation_matrix, - contact_jacobian.T))
        b_eq = - self.robot.getCoriolisAndGravityForces()

        # inequality constraints
        A_ineq = np.zeros((self.n_ineq_constraints, self.n_vars))
        b_ineq = np.zeros(self.n_ineq_constraints)
        A = np.array([[ 1, 0, 0, 0, 0, -self.d],
                      [-1, 0, 0, 0, 0, -self.d],
                      [0,  1, 0, 0, 0, -self.d],
                      [0, -1, 0, 0, 0, -self.d],
                      [0, 0, 0,  1, 0, -self.µ],
                      [0, 0, 0, -1, 0, -self.µ],
                      [0, 0, 0, 0,  1, -self.µ],
                      [0, 0, 0, 0, -1, -self.µ]])
        A_ineq[0:self.n_ineq_constraints, f_c_indices] = block_diag(A, A)

        # solve the QP, compute torques and return them
        self.qp_solver.set_values(H, F, A_eq, b_eq, A_ineq, b_ineq)
        solution = self.qp_solver.solve()
        tau = solution[tau_indices]
        return tau[6:]