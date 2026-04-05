import numpy as np

def batch_muon_directions(rng,N): # genera la dirección en la que se mueven N muones
    directions = np.empty((N,3))
    filled = 0
    while filled < N: # rejection sampling para el cos^2 con theta
        size = 2*(N-filled) # hace siempre el doble porque ese cómputo es menos costoso que repetir el loop
        theta = rng.uniform(0,np.pi/2,size)
        accept = rng.uniform(0,1,size) < np.cos(theta)**2 # chequea todos de una, más rápido
        theta = theta[accept]
        
        #phi = rng.uniform(0,2*np.pi,len(theta)) # samplea phi de una uniforme
        phi = 0
        sin_theta = np.sin(theta) # lo calculo una sola vez en lugar de 2
        new_dirs = np.stack([
            sin_theta*np.cos(phi),
            sin_theta*np.sin(phi),
            -np.cos(theta) # el vector apunta para el piso
        ],axis=1)
        
        to_add = min(N-filled,len(new_dirs)) # agrega las direcciones necesarias
        directions[filled:filled+to_add] = new_dirs[:to_add]
        filled += to_add
    
    print(theta.shape)
    return directions,theta

def batch_generate_muon_t(det_size,det_delta,det_height,margin,rate,rng,N):
    x_length = det_size # la longitud del plano en la dirección en la que no roto es el lado de los centelladores
    y_length = det_delta+2*det_height+2*margin # en la dirección en la que roto es la longitud ocupada por los centelladores
                                               # (distancia + alturas) sumando un margen a elección
    area = x_length*y_length
    
    ts = rng.exponential(scale=1/(rate*area),size=N) # sampleo el tiempo entre muones de una exponencial
    
    x_o = rng.uniform(-x_length/2,x_length/2,size=N)
    y_o = rng.uniform(-y_length/2,y_length/2,size=N)
    z_o = np.full(N,det_height+det_delta/2) # la altura de generación es el punto medio entre los centelladores
    starts = np.stack([x_o,y_o,z_o],axis=1) 

    flechas,thetas = batch_muon_directions(rng,N)
    print(thetas.shape)
    return ts,starts,flechas,thetas

def generar_lluvia(mode,det_size,det_delta,det_height,margin,seed=0,rate=1/60,t_f=0,n=0,buffer_extra=1000):
    rng = np.random.default_rng(seed) # defino el generador de números aleatorios

    if mode == 'n':  # si quiero un número de muones fijo
        ts,muon_starts,muon_flechas,thetas = batch_generate_muon_t(det_size,det_delta,det_height,margin,rate,rng,n)
        muon_times = np.cumsum(ts)

    # ACLARACIÓN: EL MODO TEMPORAL PUEDE GENERAR UN TIEMPO MENOR AL PEDIDO, PERO ES LA MANERA DE HACERLO EFICIENTE
    # SE PUEDE AUMENTAR EL buffer_extra PARA MEJORARLO PERO LA FORMA DE HACERLO CORRECTAMENTE ES CON UN LOOP MUY INEFICIENTE
    elif mode == 't':  # si quiero un tiempo de simulación fijo
        buffer_size = int(rate*det_size*(det_delta+2*det_height+2*margin)*t_f)+buffer_extra
        # sobresampleo para intentar tener suficientes
        ts,starts,flechas = batch_generate_muon_t(det_size,det_delta,det_height,margin,rate,rng,buffer_size)
        cum_ts = np.cumsum(ts)
        valid = cum_ts < t_f

        muon_times = cum_ts[valid]
        muon_starts = starts[valid]
        muon_flechas = flechas[valid]

    else:
        raise ValueError("Modo incorrecto: use 'n' (número) o 't' (tiempo)")

    print(thetas.shape)
    return muon_times,muon_starts,muon_flechas,thetas

def rotate_x(element,angle): # rota un elemento (nos va a interesar el centro de los cuadrados del detector) en un ángulo
                             # alrededor del eje x
    sen = np.sin(angle)
    cos = np.cos(angle)

    x,y,z = element.T
    rotated = np.stack([x,cos*y-sen*z,sen*y+cos*z],axis=-1)
    return rotated # calcula la matriz de rotación y se la aplica al vector (que tiene que ser un array) sin tener que pasar
                   # por el producto matricial

normal_angle = lambda ang : np.array([0.0,np.sin(ang),np.cos(ang)]) # genera el vector normal al cuadrado de los detectores
                                                                    # dado que rotaron un cierto ángulo desde la posición
                                                                    # vertical

def ray_plane_intersection(start,flecha,center,normal): # calcula el punto en el que la traza de un muón intersecta el plano
                                                        # que contiene a algún cuadrado de los detectores
    denom = np.dot(normal,flecha)
    
    if np.abs(denom) < 1e-6: # evita casos límites que son complicados numéricamente
        return None
    
    t = np.dot(normal,center-start)/denom # la distancia recorrida desde el punto start hasta el de intersección
    return start+t*flecha # devuelve el punto en el que ocurre la intersección

def calc_uv(normal): # calcula unos vectores de proyecciones que sirven para ver si un punto está adentro del cuadrado
    u = np.cross(normal,np.array([0,0,1]))
    if np.linalg.norm(u) < 1e-6:
        u = np.cross(normal,np.array([0,1,0]))
    u = u/np.linalg.norm(u)
    v = np.cross(normal,u)
    return u,v

def is_point_in_square(point,center,normal,u,v,half_len):
    local_point = point - center
    u_dist = np.dot(local_point,u)
    v_dist = np.dot(local_point,v)

    epsilon = 1e-10
    return (-half_len - epsilon <= u_dist <= half_len + epsilon) and (-half_len - epsilon <= v_dist <= half_len + epsilon)

def intersect_square(start,flecha,center,normal,u,v,half_len):
    point = ray_plane_intersection(start,flecha,center,normal)
    if point is None:
        return False
    return is_point_in_square(point,center,normal,u,v,half_len)

def intersect_detector(start,flecha,center_top,center_bottom,normal,u,v,half_len):
    top_intersection = intersect_square(start,flecha,center_top,normal,u,v,half_len)
    bottom_intersection = intersect_square(start,flecha,center_bottom,normal,u,v,half_len)
    return top_intersection or bottom_intersection

def sim_rotados(lluvia,det_size,det_height,det_xy,det_delta,freq,seed=0):
    rng = np.random.default_rng(seed) # defino el generador de números aleatorios

    ang0 = rng.uniform(0,2*np.pi) # calculo el ángulo en el que se encuentra el par de centelladores respecto al eje z
    ang0 = np.pi/4
    center_top = np.array([*det_xy,det_delta + det_height])
    center_top_up = center_top + np.array([0,0,det_height])
    center_bottom = center_top - np.array([0,0,det_delta])
    center_bottom_down = center_bottom - np.array([0,0,det_height])
    # defino el centro de los cuadrados de cada cara (superior e inferior) de cada centellador
    
    measured_angles = []
    thetas_d = []
    times,starts,flechas,thetas = lluvia

    for t,start,flecha,theta in zip(times,starts,flechas,thetas):
        ang = (np.pi*freq*t+ang0)%(2*np.pi)
        
        normal_new = normal_angle(ang) # calculo la nueva normal, la misma para todos los cuadrados
        u,v = calc_uv(normal_new) # calculo las proyecciones necesarias para las intersecciones

        # roto los centros del primer detector
        center_top_new = rotate_x(center_top,ang)
        center_top_up_new = rotate_x(center_top_up,ang)

        if intersect_detector(start,flecha,center_top_up_new,center_top_new,normal_new,u,v,det_size/2):
            # si pasó por el primer detector, roto los centros del segundo (no es necesario chequear los que ya no pasan por
            # el primero porque no se cuentan como detecciones si no pasan por ambos)
            center_bottom_new = rotate_x(center_bottom,ang)
            center_bottom_down_new = rotate_x(center_bottom_down,ang)
            if intersect_detector(start,flecha,center_bottom_new,center_bottom_down_new,normal_new,u,v,det_size/2):
                measured_angles.append(ang)
                thetas_d.append(theta)
                # si pasaron por ambos detectores, registro el ángulo en el que fueron detectados)

    return measured_angles,len(times),thetas_d

dist_centros = 24.7 # todas las distancias en cm
det_height = 1.1
det_size = 4.0
det_delta = dist_centros - det_height
det_xy = [0,0]
freq = 0 # vueltas por segunda
freqs = np.array([0,1,5,10,1e-1,5*1e-1,1e-2,5*1e-2,1e-3,5*1e-3,1e-4,5*1e-4])

lluvia = generar_lluvia('n',det_size,det_delta,det_height,det_delta/2,seed=0,n=int(1e6))
#for freq in freqs:
#    measured_angles,total_muons,thetas = sim_rotados(lluvia,det_size,det_height,det_xy,det_delta,freq,seed=0)
#    data = np.array([total_muons,*measured_angles,*thetas])
#    np.savetxt(f'freqs/{freq}.csv',data,delimiter=',')

measured_angles,total_muons,thetas = sim_rotados(lluvia,det_size,det_height,det_xy,det_delta,freq,seed=0)

data = np.array([total_muons,*measured_angles,*thetas])
#data = np.array(thetas)
np.savetxt('data_quieto_45_2_rot.csv',data,delimiter=',')
