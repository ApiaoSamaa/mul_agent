# %%
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import os
import numpy as np
from scipy.spatial.distance import pdist, squareform
import math
import streamlit as st


# Initialize plot
L = 450
fig = plt.figure()
ax = plt.axes(xlim=(-L, L), ylim=(-L, L))
points, = ax.plot([], [], '.', color = 'black')
plt.axis('off')


tsteps         = 200        # Time steps
n              = 300        # Number of birds
V0             = 20         # Initial velocity

wall_repulsion = 20         
margin         = 20
max_speed      = 20
neighbors_dist = 70         # Community distance (At which distance they apply rule 1 and rule 3)

# Rule 1: COHESION
R = 0.1                     # velocity to center contribution

# Rule 2: SEPARATION
bird_repulsion = 7          # low values  make more "stains" of birds, nice visualization
privacy        = 14         #  Avoid each other ,length. When they see each other at this distance, 

# Rule 3: ALIGNMENT
match_velocity = 3    

'''
x, y 代表 n只鸟在tsteps时候的 xy 轴坐标
''' 
x = np.zeros((n, tsteps))
y = np.zeros((n, tsteps))

x[:,0] = np.random.uniform(low=-L, high=L, size=(int(n),))
y[:,0] = np.random.uniform(low=-L, high=L, size=(int(n),))

# Randomize initial velocity
x[:,1] = x[:,0] + np.random.uniform(low=-V0, high=V0, size=(int(n),))
y[:,1] = y[:,0] + np.random.uniform(low=-V0, high=V0, size=(int(n),))

# Cohesion
def moveToCenter(x0, y0, neighbors_dist, n, R):
    '''
    此处的x0, y0 代表 0 时刻 n 只鸟的坐标，因此 x0和y0的大小为 (n,)
    '''
    # pdist Y : ndarray
    # Returns a condensed distance matrix Y. For each i and j
    # (where i<j<m),where m is the number of original observations. The metric dist(u=X[i], v=X[j]) is computed and stored in entry `m
    # i + j - ((i + 2) * (i + 1)) // 2`.
    # Convert a vector-form distance vector to a square-form distance matrix, and vice-versa.
    
    # m 使用transpose也是为了能够让不同的鸟之间计算距离。此处很方便，因为x和y刚好代表距离能够直接用内置解决
    m = squareform(pdist(np.transpose([x0, y0])))
    # 每一行都是一只鸟，看True False判断它和哪些鸟近，加入这个group
    idx = (m<neighbors_dist)
    center_x = np.zeros(n)
    center_y = np.zeros(n)
    vx = np.zeros(n)
    vy = np.zeros(n)
    for i in range(0, n-1):
        # 鸟i的 x center. 以某只鸟为中心看它周围，而不是先分出集群再来看里面的鸟的velocity
        center_x[i] = np.mean(x0[idx[i,]])
        # 鸟i的 y center
        center_y[i] = np.mean(y0[idx[i]])
        
        # 根据和中心点的距离来计算相应的动量
        vx[i] = -(x0[i] - center_x[i]) * R
        vy[i] = -(y0[i] - center_y[i]) * R
        
    return vx, vy

def avoidOthers(x0, y0, n, privacy, bird_repulsion):
    # 注意此处的 transpose! np在使用vector进行组装时候，不管怎样都按照了行堆积？
    dist = squareform(pdist(np.transpose([x0,y0])))
    """
    imat = np.array([[True, False, False], [True, False, False], [False, True, False]])
    k = np.where(imat)
    >> (array([0, 1, 2]), array([0, 0, 1])), 注意 output ⚠️ 是横纵坐标
    """ 
    
    # 会生成 和dist 大小相同的np.array
    idxmat = (dist<privacy) & (dist != 0)
    
    # 找出了哪几对是挨得近的，格式是 (num_of_pairs, 2), 其中2中内容就是那两个节点
    idx = np.transpose(np.array(np.where(idxmat)))
    
    vx = np.zeros(n)
    vy = np.zeros(n)
    
    vx[idx[:,0]] = (x0[idx[:,0]] - x0[idx[:,1]]) * bird_repulsion
    vy[idx[:,0]] = (y0[idx[:,0]] - y0[idx[:,1]]) * bird_repulsion
    return vx, vy
    
def matchVelocities(x_prev, y_prev, x0, y0, n, neighbors_dist, match_velocity): 
    # x_prev, y_prev 意思是 previous x and y， 对于每个boid需要计算其邻居的平均速度并且返回加权的match_velocity
    m = squareform(pdist(np.transpose([x_prev, y_prev])))  
    # 小于neighbors_dist 的鸟就是邻居，其中neighbors_dist 是float. idx.shape = [n, n]
    idx = (m<=neighbors_dist)
    
    vmeans_x = np.zeros(n)
    vmeans_y = np.zeros(n)
    for i in range(0, n-1):
        vmeans_x[i] = np.mean(x0[idx[i,:]] - x_prev[idx[i,]] )
        vmeans_y[i] = np.mean(y0[idx[i,:]] - y_prev[idx[i,]])
    
    return vmeans_x*match_velocity, vmeans_y*match_velocity
    
def move(x0,y0, x_prev, y_prev, n, neighbors_dist, R, privacy, bird_repulsion, match_velocity, L, margin, wall_repulsion, max_speed):
    
    vx1,vy1 = moveToCenter(x0,y0, neighbors_dist, n, R)
    vx2,vy2 = avoidOthers(x0,y0,  n, privacy, bird_repulsion)
    vx3,vy3 = matchVelocities(x_prev, y_prev, x0,y0,  n, neighbors_dist, match_velocity)

    vx = x0-x_prev + vx1 + vx2 + vx3 
    vy = y0-y_prev + vy1 + vy2 + vy3     
    
    # max speed limit
    # Matrix 2xn. Get the length of the velocity vector for each boid, and 
    # scale it with the maximum value
    v_norm = np.zeros((2,n))
    v_vector = np.array([vx,vy])
    norm = np.linalg.norm(v_vector, axis=0)
    v_norm[:, norm!=0] =  v_vector[:, norm!=0]/norm[norm!=0]*max_speed
    
    vx = v_norm[0,:]
    vy = v_norm[1,:]
    
    # Dump velocity when hits a wall    
    right_border_dist  = L - x0
    left_border_dist   = x0 + L
    upper_border_dist  = L - y0
    bottom_border_dist = y0 + L
    
    vx[right_border_dist < margin] = vx[right_border_dist < margin] - wall_repulsion
    vx[left_border_dist < margin] = vx[left_border_dist < margin] + wall_repulsion
    vy[upper_border_dist < margin] = vy[upper_border_dist < margin] - wall_repulsion
    vy[bottom_border_dist < margin] = vy[bottom_border_dist < margin] + wall_repulsion
    
    x1 = x0 + vx
    y1 = y0 + vy
    
    x1 = np.round(x1)
    y1 = np.round(y1)
    
    return x1,y1
    

for t in range(1,tsteps-1):
    x[:, t+1],y[:, t+1] = move(x[:, t],y[:, t], x[:,t-1],y[:,t-1],
                               n, neighbors_dist, R, privacy, bird_repulsion, match_velocity, 
                               L, margin, wall_repulsion, max_speed)
    
def init():
    points.set_data([], [])
    return points,

def animate(i):
    xx = x[:,i]
    yy = y[:,i]
    points.set_data(xx, yy)
    print(xx, yy)
    return points,





anim = FuncAnimation(fig, animate, init_func=init,
                                frames=tsteps-2, interval=80, blit=True)
anim.save('boid.mp4')


# # %%

# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation


# def update_line(num, data, line):
#     line.set_data(data[..., :num])
#     return line,
    
# fig1 = plt.figure()

# # Fixing random state for reproducibility
# np.random.seed(19680801)

# data = np.random.rand(2, 25)
# l, = plt.plot([], [], 'r-')
# plt.xlim(0, 1)
# plt.ylim(0, 1)
# plt.xlabel('x')
# plt.title('test')
# line_ani = animation.FuncAnimation(fig1, update_line, 25, fargs=(data, l),
#                                    interval=50, blit=True)

# line_ani.save('lines.mp4')
# # To save the animation, use the command: line_ani.save('lines.mp4')

# fig2 = plt.figure()

# x = np.arange(-9, 10)
# y = np.arange(-9, 10).reshape(-1, 1)
# base = np.hypot(x, y)
# ims = []
# for add in np.arange(15):
#     ims.append((plt.pcolor(x, y, base + add, norm=plt.Normalize(0, 30)),))

# im_ani = animation.ArtistAnimation(fig2, ims, interval=50, repeat_delay=3000,
#                                    blit=True)
# # To save this second animation with some metadata, use the following command:
# im_ani.save('im.mp4', metadata={'artist':'Guido'})

# plt.show()
    
    
    
    


# %%

import numpy as np

def ltm_construction(adj_matrix, threshold):
    n = adj_matrix.shape[0]
    active = np.zeros(n)
    active[0] = 1
    while True:
        new_active = np.zeros(n)
        for i in range(n):
            if active[i] == 0 and np.sum(active * adj_matrix[i]) >= threshold:
                new_active[i] = 1
        if np.sum(new_active) == 0:
            break
        active = np.logical_or(active, new_active)
    return active

adj_matrix = np.array([[0, 1, 1, 0, 0],
                       [1, 0, 1, 0, 0],
                       [1, 1, 0, 1, 0],
                       [0, 0, 1, 0, 1],
                       [0, 0, 0, 1, 0]])
threshold = 2

print(ltm_construction(adj_matrix, threshold))


# %%

