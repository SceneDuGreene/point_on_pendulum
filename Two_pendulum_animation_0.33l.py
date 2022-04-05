# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 10:58:09 2022

@author: mike3
"""
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import animation
from numpy import linalg as LA

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = plt.axes(xlim=(-1.75, 1.75), ylim=(-1.25, 1.25)) #set axes limits
ax.set_aspect('equal') #this will ensure that circle looks circular (as opposed to elliptical)

leg1, = ax.plot([], [],'k', lw=2) #leg of pendulum1
leg2, = ax.plot([], [],'g', lw=2) #leg of pendulum2
object1, = ax.plot([], [], lw=2) #
object2, = ax.plot([], [], lw=2)
centroid1,  = ax.plot([], [],'ko', lw=100) #centroid1 tracking
centroid2,  = ax.plot([], [],'ko', lw=100) #centroid2 tracking
centroid1_path,  = ax.plot([], [],'k', lw=2) #centroid1 path linear vel
time_tracker, = ax.plot([], [],'r', lw=2) #track time on t axis
amp_tracker, = ax.plot([], [],'r', lw=2) #track time on amp axis
point1,  = ax.plot([], [],'bo', lw=100) #point1 -> point on circle1
point1_path, = ax.plot([], [],'b', lw=2) #particle1 path
point2,  = ax.plot([], [],'mo', lw=100) #point2 -> point on circle2
point2_path, = ax.plot([], [],'m', lw=2) #particle2 path


centroid1_path_x = []; centroid1_path_y = [] #initialize centroid_path storage
point1_path_x = [] ; point1_path_y = [] #initialize track point path storage
point2_path_x = [] ; point2_path_y = [] #initialize track point path storage

# initialization function: plot the background of each frame
def init():
    leg1.set_data([], []) #leg1 -> leg of pendulum
    leg2.set_data([], []) #leg1 -> leg of pendulum
    object1.set_data([], []) #object1 -> points on circle
    object2.set_data([], []) #object2 -> follow singular point
    centroid1.set_data([], []) #centroid1 point
    centroid2.set_data([], []) #centroid2 point    
    centroid1_path.set_data([], []) 
    time_tracker.set_data([], []) 
    amp_tracker.set_data([], []) 
    point1.set_data([], []) 
    point1_path.set_data([], []) 
    point2.set_data([], []) 
    point2_path.set_data([], []) 

    return leg1, leg2, object1, object2, centroid1, centroid2, centroid1_path, time_tracker,amp_tracker, point1, point1_path,  point2, point2_path

# animation function.  This is called sequentially
def animate(i):
# Calculate and animate movemnts
    x1, y1, x1_0, y1_0, d_x1, d_y1 = pend_anim_theta(radius1,i,x10,y10,theta_vec,res)  #r between 0->1
    x2, y2, x2_0, y2_0, d_x2, d_y2 = pend_anim_theta(radius2,i,x20,y20,theta_vec_scaled,res) #r between 0->1
    r1dot = np.array([[x1_0 + d_x1],[y1_0 + d_y1]]) #moving origin1
    r2dot = np.array([[x2_0 + d_x2],[y2_0 + d_y2]]) #moving origin2
    centroid1_x = r1dot[0].item() ; centroid1_y = r1dot[1].item() #centroid point
    centroid2_x = r2dot[0].item() ; centroid2_y = r2dot[1].item() #centroid point
    centroid1_path_x.append(r1dot[0].item());centroid1_path_y.append(r1dot[1].item()) #path 
    
# Tracing - follow moving point on circle path
    theta_1 = 0 # Starting orientation
    trace_point = 0*i*int((np.floor(res/frames))) # cycle through various points on the circle (0->100)
    # Will error if above number of frames (theta, linspace<100)
    mat1 = np.array([[np.cos(theta_1),-np.sin(theta_1)],[np.sin(theta_1),np.cos(theta_1)]])
    p1 = r1dot + np.dot(mat1,np.array([[x1[trace_point]-(r1dot[0].item())],[y1[trace_point]-(r1dot[1].item())]])) #constant velocity
    px1= p1[0] ; py1=p1[1]
    point1_path_x.append(p1[0]); point1_path_y.append(p1[1]) #path 

#Rotation - follow stationary point on circle
    omega_2 = 0*np.pi/4 # Angular rate [rad/s]?
    theta_2_dot = 0*omega_2*i/3 # Starting anguar vel
    rotating_point = x2.argmax()  # choose a point on circle to follow (0->100)
    mat2 = np.array([[np.cos(theta_2_dot),-np.sin(theta_2_dot)],[np.sin(theta_2_dot),np.cos(theta_2_dot)]])
    p2 = r2dot + np.dot(mat2,np.array([[x2[rotating_point]-(r2dot[0].item())],[y2[rotating_point]-(r2dot[1].item())]])) #constant velocity
    px2= p2[0] ; py2=p2[1]
    point2_path_x.append(p2[0]); point2_path_y.append(p2[1]) #path 
    
# Leg of Pendulum1
    leg1x1 = [x1_0,centroid1_x] ; leg1y1 = [y1_0,centroid1_y] #leg depends on radius1
    leg1_len = round(LA.norm([leg1x1[0]-leg1x1[1],leg1y1[0]-leg1y1[1]]),3) #
    # find Euclidean distance

# Leg of Pendulum2
    leg2x1 = [x2_0,centroid2_x] ; leg2y1 = [y2_0,centroid2_y] #leg depends on radius1
    leg2_len = round(LA.norm([leg2x1[0]-leg2x1[1],leg2y1[0]-leg2y1[1]]),3) #
    # find Euclidean distance
    
    ax.set_title("Kinematic Pend1, Pend2 lengths = {},{}".format(leg1_len,leg2_len))


    n = 5 #length of chemtrail
    if i<n:
        leg1.set_data(leg1x1, leg1y1) #leg1 -> leg of pendulum
        leg2.set_data(leg2x1, leg2y1) #leg1 -> leg of pendulum    
        object1.set_data(x1, y1) #object1 -> points on circle1
        object2.set_data(x2, y2) #object2 -> points on circle2
        centroid1.set_data(centroid1_x, centroid1_y)  #centroid1 point
        centroid2.set_data(centroid2_x, centroid2_y)  #centroid2 point
        centroid1_path.set_data(centroid1_path_x, centroid1_path_y) #chemtrails
        time_tracker.set_data((centroid1_x, centroid1_x),(-1.15, -1.25))
        amp_tracker.set_data((-1.75,-1.65),(centroid1_y,centroid1_y))
        point1.set_data(px1, py1)  #object1 -> points on circle1
        point1_path.set_data(point1_path_x,point1_path_y) #point_path circle1
        point2.set_data(px2, py2)  #object2 -> points on circle2
        point2_path.set_data(point2_path_x,point2_path_y) #point_path circle2
# 

    else:
        leg1.set_data(leg1x1, leg1y1) #leg1 -> leg of pendulum
        leg2.set_data(leg2x1, leg2y1) #leg1 -> leg of pendulum            
        object1.set_data(x1, y1) #object1 -> points on circle1
        object2.set_data(x2, y2) #object2 -> points on circle2
        centroid1.set_data(centroid1_x, centroid1_y)  #centroid1 point
        centroid2.set_data(centroid2_x, centroid2_y)  #centroid2 point        
        centroid1_path.set_data(centroid1_path_x[-n*2:], centroid1_path_y[-n*2:]) #chemtrails
        time_tracker.set_data((centroid1_x, centroid1_x),(-1.15, -1.25))
        amp_tracker.set_data(((-1.75,-1.65),(centroid1_y,centroid1_y)))        
        point1.set_data(px1, py1)
        point1_path.set_data(point1_path_x[-n:],point1_path_y[-n:])
        point2.set_data(px2, py2)  #object2 -> points on circle2
        point2_path.set_data(point2_path_x[-n:],point2_path_y[-n:]) #point_path circle2

    
    return leg1, leg2, object1, object2, centroid1, centroid2, centroid1_path, time_tracker, amp_tracker, point1, point1_path, point2, point2_path


    
def simplePendulumSimulation(theta0,omega0,tau,m,g,l,numSteps,plotFlag):
    # This function simulates a simple pendulum using the Euler-Cromer method.
    # Inputs: theta0 (the initial angle, in rad)
    #          omega0 (the initial angular velocity, in rad/s)
    #          tau (the time step)
    #          m (mass of the pendulum)
    #          g (gravitational constant)
    #          l (length of the pendulum)
    #          numSteps (number of time steps to take, in s)
    #          plotFlag (set to 1 if you want plots, 0 otherwise)
    # Outputs: t_vec (the time vector)
    #           theta_vec (the angle vector)

    # initialize vectors

    time_vec = [0]*numSteps
    theta_vec = [0]*numSteps
    omega_vec = [0]*numSteps
    KE_vec = [0]*numSteps
    PE_vec = [0]*numSteps

    # set initial conditions

    theta = theta0
    omega = omega0
    time = 0

    # begin time stepping

    for i in range(0,numSteps):

            omega_old = omega
            theta_old = theta
            # update the values
            omega = omega_old - (g/l)*sin(theta_old)*tau
            theta = theta_old + omega*tau
            # record the values
            time_vec[i] = tau*i
            theta_vec[i] = round(theta,3)
            omega_vec[i] = omega
            KE_vec[i] = (1/2)*m*l**2*omega**2
            PE_vec[i] = m*g*l*(1-cos(theta))

    TE_vec = np.add(KE_vec,PE_vec)

    # make graphs

    if plotFlag == 1:

        plt.figure(0)
        plt.plot(time_vec,theta_vec)
        plt.xlabel('Time (s)')
        plt.ylabel('Displacement (rad)')
        plt.savefig('plot1.png', bbox_inches='tight')

#        plt.figure(1)
#        plt.plot(time_vec,KE_vec,label='Kinetic Energy')
#        plt.plot(time_vec,PE_vec,label='Potential Energy')
#        plt.plot(time_vec,TE_vec,label='Total Energy')
#        plt.legend(loc='upper left')
#        plt.xlabel('Time (s)')
#        plt.ylabel('Energy (J)')
#        plt.savefig('plot2.png', bbox_inches='tight')
#
#        plt.figure(2)
#        plt.plot(theta_vec,omega_vec)
#        plt.xlabel('Displacement (rad)')
#        plt.ylabel('Velocity (rad/s)')
#        plt.savefig('plot3.png', bbox_inches='tight')
#
#        plt.show()

    # return the vectors
    return theta_vec

def pend_anim_theta(r,i,x_0,y_0,thetas,res): #r between 0->1
    # theta goes from 0 to 2pi
    theta = np.linspace(0, 2*np.pi,res) 
    # the radius of the circle
    # compute cartesian x1 and x2
#    d_x = *np.sin(thetas[i]) ; d_y = -*np.cos(thetas[i]) #gravity (-y)
    d_x = r*np.sin(thetas[i]) ; d_y = -r*np.cos(thetas[i]) #rotate about pt (x10,y10)

    circlex1 = r*np.cos(theta) + x_0 + d_x 
    circley1 = r*np.sin(theta) + y_0 + d_y
    return circlex1, circley1, x_0, y_0, d_x, d_y

#Pendulum Charateristics
# Draw circles at each frame i over entire animation
res = 100 # Resolution of shape. keep between 50->200
radius1 = 1; radius2 = 0.33 #radius values

x10 = 0.0 ; y10 = 1 # initial x,y position
x20 = 0.0 ; y20 = 1 # initial x,y position

# Bob mass (kg), pendulum length (m), acceleration due to gravity (m.s-2).
m, L, g = 1, radius1, 9.81
theta0, v0 = np.radians(30), 0
# Estimate of the period using the harmonic (small displacement) approximation.
# The real period will be longer than this.
Tharm = 2 * np.pi * np.sqrt(L / g)
# Time step for numerical integration of the equation of motion (s).
dt = 0.1
frames = 20; #interval = 15 #Constants for animate function
T = frames * dt
interval = dt *10

theta_vec = simplePendulumSimulation(theta0,v0,dt,m,g,radius1,frames,0)
theta_vec_scaled = simplePendulumSimulation(theta0,v0,dt,m,g,radius2,frames,0)
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=frames, interval=interval, blit=True, repeat=False)
plt.xlabel("time (s)"); plt.ylabel(" Amplitude (m)")

#f = r"c://Users/mike3/Desktop/pendulum_animation.gif"
f = r"c://Users/mike3/OneDrive/Desktop/pendulum_theta_animation.gif"
writergif = animation.PillowWriter(fps=15)
anim.save(f, writer=writergif)

plt.show()

