##################################################
################### Chase code ###################
##################################################

# packages
import numpy as np
from numpy import pi


# rotation matrix
def rot_mat(th):
    return np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
# rotate 90 degrees counterclockwise
def rot_90(x):
    return np.array([-x[1],x[0]])


##################################################
#################### Controls ####################
##################################################

# tracking control schemes
def uCATD(nu=0.6,mu=10):
    def u(t,rp,re,xp,xe):
        r = rp-re
        rd = xp-nu*xe
        u_r = r/np.linalg.norm(r)
        return -mu * u_r.dot(rot_90(rd))
    return u

def ubearing(th=0,nu=0.6,eta=10):
    def u(t,rp,re,xp,xe):
        R = rot_mat(th)
        r = rp-re
        rd = xp-nu*xe
        u_r = r/np.linalg.norm(r)
        return -u_r.dot(eta*R@rot_90(xp)+rot_90(rd)/np.linalg.norm(r))
    return u


# evader control schemes (mostly ignores state
def uconst(c=0.6):
    def u(t,rp,re,xp,xe):
        return c
    return u

def ucos(a=1,w=1,w0=0):
    def u(t,rp,re,xp,xe,nu=0.6):
        return a*np.cos(w*t+w0)
    return u


##################################################
############### Tracking procedure ###############
##################################################

# Inputs:
# - up/ue = predator/evader controllers, with inputs as positions and first derivatives
# Optional inputs:
# - rp0/reo = predator/evader initial position
# - thp0/the0 = predator/evader initial direction
# - nu = evader speed
# - T = final time
# - eps = catching condition
def Chase(up, ue, nu = 0.6, umax = 3,
          rp0 = np.array([-5,0]), re0 = np.zeros(2),
          T = 10, dt = 1e-3, eps = 1e-2, solver = 'trapezoidal',
          **kwargs):
    
    # random initial angles
    # uses kwargs to make random value change
    if 'th_e0' in kwargs: th_e0 = kwargs['th_e0']
    else: th_e0 = np.random.random()*2*pi
    if 'th_p0' in kwargs: th_p0 = kwargs['th_p0']
    else: th_p0 = np.random.random()*2*pi
        
    # time array
    t = np.arange(0,T+dt,dt)
    # position arrays
    rp = np.zeros((len(t),2))
    rp[0] = rp0
    re = np.zeros_like(rp)
    re[0] = re0
    # current angles and directions of particles
    th_p = np.zeros(len(t))
    th_e = np.zeros_like(th_p)
    th_p[0] = th_p0
    th_e[0] = th_e0
    
    # time update
    for i in range(1,len(t)):
        # explicit trapezoidal rule
        
        # initial velocities
        xp0 = np.array([np.cos(th_p[i-1]),np.sin(th_p[i-1])])
        xe0 = np.array([np.cos(th_e[i-1]),np.sin(th_e[i-1])])
        
        # forward euler
        if solver == 'euler':
            # update poistions
            rp[i] = rp[i-1] +    dt*xp0
            re[i] = re[i-1] + nu*dt*xe0

            # update angles
            th_p[i] = th_p[i-1] +    dt*min(up(t[i-1],rp[i-1],re[i-1],xp0,xe0),umax)
            th_e[i] = th_e[i-1] + nu*dt*min(ue(t[i-1],rp[i-1],re[i-1],xp0,xe0),umax)
        
        # explicit trapezoidal
        if solver == 'trapezoidal':
            # initial angle velocities
            dth_p0 = min(up(t[i-1], rp[i-1], re[i-1], xp0, xe0),umax)
            dth_e0 = min(ue(t[i-1], rp[i-1], re[i-1], xp0, xe0),umax)

            # secondary velocities
            xp1 = np.array([np.cos(th_p[i-1]+   dt*dth_p0),np.sin(th_p[i-1]+   dt*dth_p0)])
            xe1 = np.array([np.cos(th_e[i-1]+nu*dt*dth_e0),np.sin(th_e[i-1]+nu*dt*dth_e0)])

            # secondary angle velocities
            dth_p1 = min(up(t[i], rp[i-1]+dt*xp0, re[i-1]+nu*dt*xe0, xp1, xe1),umax)
            dth_e1 = min(ue(t[i], rp[i-1]+dt*xp0, re[i-1]+nu*dt*xe0, xp1, xe1),umax)

            # position update
            rp[i] = rp[i-1] +    dt*(xp0+xp1)/2
            re[i] = re[i-1] + nu*dt*(xe0+xe1)/2

            # angle update
            th_p[i] = th_p[i-1] +    dt*(dth_p0+dth_p1)/2
            th_e[i] = th_e[i-1] + nu*dt*(dth_e0+dth_e1)/2
        
        if np.linalg.norm(rp[i]-re[i]) < eps:
            return t[:(i+1)], rp[:(i+1)], re[:(i+1)], th_p[:(i+1)], th_e[:(i+1)]
        
    # return with entire trajectory
    return t,rp,re,th_p,th_e

