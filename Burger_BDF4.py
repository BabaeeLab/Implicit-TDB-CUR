#%%
import numpy as np
import scipy as sp
from scipy import linalg as la
from scipy.sparse import diags
from scipy.sparse.linalg import gmres
from tqdm import trange
import matplotlib.pyplot as plt

dt = 0.01; T = 1 # Time step and final time

ns = 2**5 # Number of samples

p = 15 #  Number of oversampling points

eps_l, eps_u = 1e-9, 1e-8 # Lower and upper threshold for rank adaptation

eps = 1e-12 # Tolerance for Newton's method

N, nu = 2**9, 0.01 # Number of grid points and viscosity

save_iter = 2 # Save every save_iter iterations

svd = lambda A : la.svd(A, full_matrices=False)

def SVD(X, wp=1, wr=1):
    X = np.sqrt(wp) * X * np.sqrt(wr)
    U, S, YT = svd(X)
    U = U * wp**-0.5
    Y = (wr**-0.5 * YT).T 
    return U, S, Y

def Trunc(U, S, Y):
    S = S[S > (np.finfo(np.float64).eps * max(U.shape[0], Y.shape[0]) * S[0])]
    r = len(S)
    return U[:,:r], S[:r], Y[:,:r], r

def GPODE(XI, r):
    G = lambda S: S[-2]**2-S[-1]**2 if XI.shape[1]>1 else S[-1]**2

    XIT = XI.T
    I = [np.argmax(np.sum(XIT**2, axis=0))]
    q = XIT[:,[I[-1]]]/la.norm(XIT[:,I[-1]])
    for i in range(1, r):
        if i<XI.shape[1]:
            XIT = XIT - q@q.T@XIT
            I += [np.argmax(np.sum(XIT**2, axis=0))]
            q = XIT[:,[I[-1]]]/la.norm(XIT[:,I[-1]])
        else:
            _, S, WT = svd(XI[I,:])
            g = G(S)
            Ub = WT @ XI.T
            R = g + np.sum(Ub**2, 0)
            R = R - np.sqrt(R**2 - 4*g*Ub[-1,:]**2)
            R[I] = -1
            I += [np.argmax(R)]

    return np.array(I)

def Grid_FD(Xmin, Xmax, N, bc_left, bc_right):

    xp = np.linspace(Xmin, Xmax, N)

    dx = xp[1]-xp[0]

    D1 = (                -   np.eye(N, k=-1)               +   np.eye(N, k=1)                 )/(2*dx )
    D2 = (                    np.eye(N, k=-1) - 2*np.eye(N) +   np.eye(N, k=1)                 )/(dx**2)
    D4 = (np.eye(N, k=-2) - 4*np.eye(N, k=-1) + 6*np.eye(N) - 4*np.eye(N, k=1) + np.eye(N, k=2))/(dx**4)

    if bc_left  =='D': D1[ 0], D2[ 0] = 0, 0
    if bc_right =='D': D1[-1], D2[-1] = 0, 0
    if bc_left  =='P': 
        D1[0,-1] = -1/(2*dx ) 
        D2[0,-1] =  1/(dx**2) 
        D4[0,-2] =  1/(dx**4); D4[0,-1] = -4/(dx**4); D4[1,-1] = 1/(dx**4)
    if bc_right =='P':
        D1[-1,0] = 1/(2*dx )
        D2[-1,0] = 1/(dx**2)
        D4[-1,1] = 1/(dx**4); D4[-1,0] = -4/(dx**4); D4[-2,0] = 1/(dx**4)

    return D1, D2, D4, xp, dx

D1, D2, _, xp, dx = Grid_FD(0, 1, N, 'D', 'D'); D2 = nu*D2
d_D2_0 = np.diag(D2, k=1); d_D2_1 = np.diag(D2); d_D2_2 = np.diag(D2, k=-1)
D1S = sp.sparse.csc_array(D1); D2S = sp.sparse.csc_array(D2)
wp = dx*np.ones((N,1))
wr = np.full(ns,1/ns).reshape(1,-1)

u = np.zeros((4,N,ns))
u[0] = np.load("Data/Burger_IC.npy")

l_time,l_S_FOM,l_S_TDB,l_Err,l_r,l_V_TDB,basis = [[] for _ in range(7)]

RHS_FOM = lambda u: -u*(D1S@u) + D2S@u 

def Solve_FOM_IRK4(u):

    def A_FOM(u):
        A = np.zeros((3,N))
        A[0, 1:] = (1/4)*dt*( u[:-1]/(2*dx) - d_D2_0)
        A[1,  :] = (1/4)*dt*(D1S@u          - d_D2_1) + 1
        A[2,:-1] = (1/4)*dt*(-u[ 1:]/(2*dx) - d_D2_2)
        return diags([A[2,:-1], A[1,:], A[0,1:]], [-1, 0, 1])

    k1, dk1 = RHS_FOM(u), 1
    while (la.norm(dk1)/N)>eps:
        u1  = u + (1/4)*dt*k1
        dk1 = gmres(A_FOM(u1), RHS_FOM(u1)-k1, tol=1e-07)[0]
        k1 += dk1

    k2, dk2 = k1.copy(), 1
    c2 = u + (1/2)*dt*k1
    while (la.norm(dk2)/N)>eps: 
        u2 = c2 + (1/4)*dt*k2
        dk2 = gmres(A_FOM(u2), RHS_FOM(u2)-k2, tol=1e-07)[0]
        k2 += dk2

    k3, dk3 = k2.copy(), 1
    c3 = u + (17/50)*dt*k1 - (1/25)*dt*k2
    while (la.norm(dk3)/N)>eps: 
        u3 = c3 + (1/4)*dt*k3
        dk3 = gmres(A_FOM(u3), RHS_FOM(u3)-k3, tol=1e-07)[0]
        k3 += dk3

    k4, dk4 = k3.copy(), 1
    c4 = u + (371/1360)*dt*k1 - (137/2720)*dt*k2 + (15/544)*dt*k3
    while (la.norm(dk4)/N)>eps: 
        u4 = c4 + (1/4)*dt*k4
        dk4 = gmres(A_FOM(u4), RHS_FOM(u4)-k4, tol=1e-07)[0]
        k4 += dk4

    k5, dk5 = k4.copy(), 1
    c5 = u + (25/24)*dt*k1 - (49/48)*dt*k2 + (125/16)*dt*k3 - (85/12)*dt*k4
    while (la.norm(dk5)/N)>eps: 
        u5 = c5 + (1/4)*dt*k5
        dk5 = gmres(A_FOM(u5), RHS_FOM(u5)-k5, tol=1e-07)[0]
        k5 += dk5

    u = u + dt*((25/24)*k1 - (49/48)*k2 + (125/16)*k3 - (85/12)*k4 + (1/4)*k5)

    return u

def Solve_FOM_BDF4(u):
    u_k, du = u[-1].copy(), 1

    const = (48*u[3] - 36*u[2] + 16*u[1] - 3*u[0])/25

    A = np.zeros((3,N))

    while la.norm(du)>eps:

        R = const - u_k + dt*(12/25)*RHS_FOM(u_k)

        A[0, 1:] = (12/25)*dt*( u_k[:-1]/(2*dx) - d_D2_0)
        A[1,  :] = (12/25)*dt*(D1S@u_k          - d_D2_1) + 1
        A[2,:-1] = (12/25)*dt*(-u_k[ 1:]/(2*dx) - d_D2_2)
        du = gmres(sp.sparse.diags([A[2,:-1], A[1,:], A[0,1:]], [-1, 0, 1]), R, tol=1e-07)[0]

        u_k += du

    u[:3] = u[1:4].copy()
    u[-1] = u_k       

    return u

RHS_TDB_r = lambda u_Ir, u_Ire, Ir, Ire: -u_Ir*(D1[Ir][:,Ire]@u_Ire) + D2[Ir][:,Ire]@u_Ire

def Solve_TDB_r(u_Ir, u_Ire, Ir, Ire, q, U, pinv_UIr, u):

    u_k_Ir, u_k_Ire, du_k_Ir, range_Ir = u_Ir[-1].copy(), u_Ire[-1].copy(), 1, np.arange(len(Ir)) 
    u_k = u[-1].copy()

    const = (48*u_Ir[3] - 36*u_Ir[2] + 16*u_Ir[1] - 3*u_Ir[0])/25 
    c = (48*u[3] - 36*u[2] + 16*u[1] - 3*u[0])/25

    i = 0
    res = []
    while (la.norm(du_k_Ir)/len(Ir))>eps and (i<6):
        i += 1

        R_Ir = const - u_k_Ir + dt*(12/25)*RHS_TDB_r(u_k_Ir, u_k_Ire, Ir, Ire)
        res += [c - u_k + dt*(12/25)*RHS_FOM(u_k)]

        A_p_Ir_minus_1 = (12/25)*dt*(-u_k_Ir/(2*dx)        - D2[Ir,Ir-1]).reshape(-1,1)
        A_p_Ir         = (12/25)*dt*(D1[Ir][:,Ire]@u_k_Ire - D2[Ir,Ir  ]).reshape(-1,1) + 1
        A_p_Ir_plus_1  = (12/25)*dt*( u_k_Ir/(2*dx)        - D2[Ir,Ir+1]).reshape(-1,1)
        Ar = A_p_Ir_minus_1 * U[Ir-1] + A_p_Ir * U[Ir] + A_p_Ir_plus_1 * U[Ir+1]
        z = la.lstsq(Ar, R_Ir)[0]

        du_k_Ir = U[Ir] @ z
        u_k_Ir  += du_k_Ir
        u_k_Ire += U[Ire] @ z

        u_k += U@z

    u_Ir[:-1]  =  u_Ir[1:].copy();  u_Ir[-1] =  u_k_Ir.ravel()    
    u_Ire[:-1] = u_Ire[1:].copy(); u_Ire[-1] = u_k_Ire.ravel()

    return u_Ir, u_Ire, res

def Solve_TDB(U, S, Y, r, basis):

    leps = S[-1]/la.norm(S)
    if   leps>eps_u: r += 1
    elif leps<eps_l: r -= 1
    
    Ic = GPODE(Y, r)

    V_Ic = np.array([basis[i][0] @ basis[i][1][:,Ic] for i in range(4)])

    for i in range(len(Ic)): V_Ic[:,:,i] = Solve_FOM_BDF4(V_Ic[:,:,i])
    
    U_F, S_F, Y_F = SVD(np.hstack((V_Ic[-1], V_Ic[-2])), wp=wp)
    U_F, S_F, Y_F, r_F = Trunc(U_F, S_F, Y_F)

    Ir = GPODE(U_F, r_F+p); Ir = np.sort(Ir)
    q = np.setdiff1d(np.arange(N), Ir)
    Ire = np.unique(np.nonzero(D2[Ir,:])[1])

    V_Ir  = np.array([basis[i][0][Ir]  @ basis[i][1] for i in range(4)])
    V_Ire = np.array([basis[i][0][Ire] @ basis[i][1] for i in range(4)])

    pinv_U_Ic = la.pinv(U_F[Ir])
    
    V = np.array([basis[i][0]  @ basis[i][1] for i in range(4)])

    for i in range(ns): V_Ir[:,:,i], V_Ire[:,:,i], res_ = Solve_TDB_r(V_Ir[:,:,i], V_Ire[:,:,i], Ir, Ire, q, U_F, pinv_U_Ic, V[:,:,i])
    
    U = SVD(V_Ic[-1], wp=wp)[0]
    R, S, Y = SVD(la.lstsq(U[Ir,:], V_Ir[-1])[0], wr=wr)
    U = U@R

    basis[:-1] = basis[1:].copy()
    basis[ -1] = [U*S,Y.T]

    return U, S, Y, r, basis

r = 5 # Initial rank of the reduced model

for i in range(1,4): u[i] = np.apply_along_axis(Solve_FOM_IRK4, 0, u[i-1].copy())
for i in range(4):
    U, S, Y = SVD(u[i], wp, wr)
    U, S, Y = U[:,:r], S[:r], Y[:,:r]
    basis += [[U*S, Y.T]]

for iter in trange(3, round(T/dt)):
    
    for i in range(ns): u[:,:,i] = Solve_FOM_BDF4(u[:,:,i])

    U, S, Y, r, basis = Solve_TDB(U, S, Y, r, basis)

    if (iter+1)%save_iter==0: 
        l_time += [(iter+1)*dt]

        l_r  += [r]
        S_ = np.full(ns, np.NaN); S_[:S.shape[0]] = S
        l_S_TDB += [S_.copy()]

        l_S_FOM += [SVD(np.sqrt(wp)*u[-1]*np.sqrt(wr))[1]]
        l_V_TDB += [U*S@Y.T]

fig, axs = plt.subplots(1, 2, figsize=(8,4), dpi=300, constrained_layout=True)
axs[0].semilogy(l_time, np.array(l_S_FOM)[:,:max(l_r)-1], 'b-' )
axs[0].semilogy(l_time, np.array(l_S_TDB)[:,:max(l_r)-1], 'r--')
axs[0].semilogy(l_time, np.array(l_S_FOM)[:,max(l_r)], 'b-' , label='FOM')
axs[0].semilogy(l_time, np.array(l_S_TDB)[:,max(l_r)], 'r--', label='TDB-CUR')
axs[0].set_xlabel('$t$')
axs[0].set_ylabel('$\Sigma(t)$')
axs[0].grid(color='gray', linestyle='--', alpha=.75)
axs[0].legend()
xc, tc = np.meshgrid(xp, l_time)
axs[1].contourf(xc, tc, np.mean(l_V_TDB, axis=2), 100, cmap='bwr', zorder=0)
axs[1].set_xlabel('$x$')
axs[1].set_ylabel('$t$')
