import numpy as np
import pandas as pd

def single_muon_direction(rng):
    while True:
        theta = rng.uniform(0,np.pi/2)
        if rng.uniform() < np.cos(theta)**2:
            break
    phi = rng.uniform(0,2*np.pi)
    return np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),-np.cos(theta)])
    
def generate_muon(R,rng):
    theta_o = rng.uniform(0,np.pi/2)
    phi_o = rng.uniform(0,2*np.pi)
    start = R*np.array([np.sin(theta_o)*np.cos(phi_o),np.sin(theta_o)*np.sin(phi_o),np.cos(theta_o)])
    flecha = single_muon_direction(rng)
    return start,flecha

def generate_muon_t(R,rate,rng):
    t = rng.exponential(scale=1/(rate*2*np.pi*R**2))
    theta_o = rng.uniform(0,np.pi/2)
    phi_o = rng.uniform(0,2*np.pi)
    start = R*np.array([np.sin(theta_o)*np.cos(phi_o),np.sin(theta_o)*np.sin(phi_o),np.cos(theta_o)])
    flecha = single_muon_direction(rng)
    return t,start,flecha

def generar_lluvia(mode,R,seed=0,rate=1/60,t_f=0,n=0):
    rng = np.random.default_rng(seed)

    muon_paths = []
    muon_times = []

    t = 0
    
    if mode == 'n':
        while len(muon_paths) < n:
            new_t,start,flecha = generate_muon_t(R,rate,rng)
            t += new_t
            muon_times.append(t)
            muon_paths.append((start,flecha))

    elif mode == 't':
        muon_times = []
        while t < t_f:
            new_t,start,flecha = generate_muon_t(R,rate,rng)
            t += new_t
            muon_times.append(t)
            muon_paths.append((start,flecha))
        
    else:
        print('Modo incorrecto')
        return None
    return muon_times,muon_paths

def generar_lluvia_estatica(R,n,seed=0):
    rng = np.random.default_rng(seed)

    muon_paths = []

    while len(muon_paths) < n:
        start,flecha = generate_muon(R,rng)
        muon_paths.append((start,flecha))
    return muon_paths

def rotate_x(element,angle):
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(angle), -np.sin(angle)],
        [0, np.sin(angle),  np.cos(angle)]
    ])
    return element @ Rx.T

def rotate_y(element,angle):
    Ry = np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0,  np.cos(angle)]
    ])
    return element @ Ry.T

normal_angle = lambda ang : np.array([0,np.sin(ang),np.cos(ang)])

def ray_plane_intersection(start,flecha,center,normal):
    denom = np.dot(normal,flecha)
    if np.abs(denom) < 1e-6:
        return None
    t = np.dot(normal,center-start) / denom
    intersection_point = start + t * flecha
    return intersection_point

def is_point_in_square(point,center,normal,side):
    u = np.cross(normal,np.array([0, 0, 1]))
    if np.linalg.norm(u) < 1e-6:
        u = np.cross(normal, np.array([0, 1, 0]))
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)

    local_point = point - center
    u_dist = np.dot(local_point, u)
    v_dist = np.dot(local_point, v)

    half_len = side / 2
    epsilon = 1e-10
    return (-half_len - epsilon <= u_dist <= half_len + epsilon) and (-half_len - epsilon <= v_dist <= half_len + epsilon)

def intersect_square(start,flecha,center,normal,side):
    point = ray_plane_intersection(start,flecha,center,normal)
    if point is None:
        return False
    return is_point_in_square(point,center,normal,side)

def intersect_detector(start,flecha,center_top,center_bottom,normal,side):
    return intersect_square(start,flecha,center_top,normal,side) or intersect_square(start,flecha,center_bottom,normal,side)

def rotate_desc(element,center,ang,func):
    desc = element - center
    desc = func(desc,ang)
    return desc + center

def sim_3_fijos_fast(lluvia,det_size,det_height,det_xy,det_delta,det_ang):
    normal_c = normal_angle(0)
    center_top_c = np.array([*det_xy,det_delta+det_height])
    center_top_up_c = center_top_c + np.array([0,0,det_height])
    center_bottom_c = center_top_c - np.array([0,0,det_delta])
    center_bottom_down_c = center_bottom_c - np.array([0,0,det_height])


    normal_i = normal_angle(-det_ang)
    center_top_i = np.array([*det_xy,det_delta+det_height]) - np.array([det_size,0,0])
    center_top_up_i = center_top_i + np.array([0,0,det_height])
    center_bottom_i = center_bottom_c + np.array([det_size,0,0])
    center_bottom_down_i = center_bottom_i - np.array([0,0,det_height])

    center_top_i = rotate_desc(center_top_i,center_top_i+np.array([det_size/2,-det_size/2,0]),-det_ang,rotate_y)
    center_top_up_i = rotate_desc(center_top_up_i,center_top_i+np.array([det_size/2,-det_size/2,0]),-det_ang,rotate_y)
    center_bottom_i = rotate_desc(center_bottom_i,center_bottom_i+np.array([-det_size/2,-det_size/2,0]),-det_ang,rotate_y)
    center_bottom_down_i = rotate_desc(center_bottom_down_i,center_bottom_i+np.array([-det_size/2,-det_size/2,0]),-det_ang,rotate_y)


    normal_d = normal_angle(det_ang)
    center_top_d = np.array([*det_xy,det_delta+det_height]) + np.array([det_size,0,0])
    center_top_up_d = center_top_d + np.array([0,0,det_height])
    center_bottom_d = center_bottom_c - np.array([det_size,0,0])
    center_bottom_down_d = center_bottom_d - np.array([0,0,det_height])

    center_top_d = rotate_desc(center_top_d,center_top_d+np.array([-det_size/2,-det_size/2,0]),det_ang,rotate_y)
    center_top_up_d = rotate_desc(center_top_up_d,center_top_d+np.array([-det_size/2,-det_size/2,0]),det_ang,rotate_y)
    center_bottom_d = rotate_desc(center_bottom_d,center_bottom_d+np.array([det_size/2,-det_size/2,0]),det_ang,rotate_y)
    center_bottom_down_d = rotate_desc(center_bottom_down_d,center_bottom_d+np.array([det_size/2,-det_size/2,0]),det_ang,rotate_y)
    
    muon_det = []

    for vectors in lluvia:
        start,flecha = vectors

        if intersect_detector(start,flecha,center_top_up_c,center_top_c,normal_c,det_size) and intersect_detector(start,flecha,center_bottom_c,center_bottom_down_c,normal_c,det_size):
            muon_det.append(0)
        elif intersect_detector(start,flecha,center_top_up_i,center_top_i,normal_i,det_size) and intersect_detector(start,flecha,center_bottom_i,center_bottom_down_i,normal_i,det_size):
            muon_det.append(1)
        elif intersect_detector(start,flecha,center_top_up_d,center_top_d,normal_d,det_size) and intersect_detector(start,flecha,center_bottom_d,center_bottom_down_d,normal_d,det_size):
            muon_det.append(2)
    return muon_det.count(0),muon_det.count(1),muon_det.count(2)


lluvia_estatica = generar_lluvia_estatica(10,300000,seed=1)
alphas = np.pi*np.arange(20,46)/180
det_i = []
det_d = []
det_c = []
for alpha in alphas:
    det0,det1,det2 = sim_3_fijos_fast(lluvia_estatica,4,0.5,[0,0],20,alpha)
    det_c.append(det0)
    det_i.append(det1)
    det_d.append(det2)

df = pd.DataFrame([det_i,det_d,det_c])
df.to_csv('detecciones_3_fijos.csv')
