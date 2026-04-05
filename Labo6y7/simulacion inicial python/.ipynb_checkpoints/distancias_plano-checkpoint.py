import numpy as np

def batch_muon_directions(rng,N): # genera la dirección en la que se mueven N muones
    directions = np.empty((N,3)) # inicializa el array a llenar
    filled = 0
    while filled < N: # rejection sampling para el cos^2 con theta
        size = 2*(N-filled) # hace siempre el doble porque ese cómputo es menos costoso que repetir el loop
        theta = rng.uniform(0,np.pi/2,size)
        accept = rng.uniform(0,1,size) < np.cos(theta)**2 # chequea todos de una, más rápido
        theta = theta[accept]

#        phi = rng.uniform(0,2*np.pi,len(theta)) # samplea phi de una uniforme
        # estas dos líneas son si se quiere hacer 2D y que salga desde los dos sentidos y no de un solo lado
        signo = rng.integers(0,2,len(theta))
        phi = np.pi/2 + signo*np.pi
        sin_theta = np.sin(theta) # lo calculo una sola vez en lugar de 2
        new_dirs = np.stack([
            sin_theta*np.cos(phi),
            sin_theta*np.sin(phi),
            -np.cos(theta) # el vector apunta para el piso
        ],axis=1)
        
        to_add = min(N-filled,len(new_dirs)) # agrega las direcciones necesarias
        directions[filled:filled+to_add] = new_dirs[:to_add]
        filled += to_add
    
    return directions

def batch_generate_muon_t(det_size,det_delta,det_height,margin,rate,rng,N):
    x_length = det_size # la longitud del plano en la dirección en la que no roto es el lado de los centelladores
    y_length = det_delta+2*det_height+2*margin # en la dirección en la que roto es la longitud ocupada por los centelladores (distancia + alturas) sumando un margen a elección
    area = x_length*y_length
    
    ts = rng.exponential(scale=1/(rate*area),size=N) # sampleo el tiempo entre muones de una exponencial
    ts = np.cumsum(ts) # como samplear de la exponencial me da los interavlos entre muones, los sumo para tener los tiempos absolutos
    
    x_o = rng.uniform(-x_length/2,x_length/2,size=N) # sampleo de una uniforme 2D en el plano de generación
    y_o = rng.uniform(-y_length/2,y_length/2,size=N)
    z0 = det_height+det_delta/2
    z_o = np.full(N,z0) # la altura de generación es el punto medio entre los centelladores
    starts = np.stack([x_o,y_o,z_o],axis=1) # junto todo en un array de N vectores de 3 componentes

    flechas = batch_muon_directions(rng,N)
    return ts,starts,flechas

def generar_lluvia(det_size,det_delta,det_height,n,margin=0,seed=0,rate=1/60):
    rng = np.random.default_rng(seed) # defino el generador de números aleatorios

    muon_times,muon_starts,muon_flechas = batch_generate_muon_t(det_size,det_delta,det_height,margin,rate,rng,n)

    print(np.mean(np.diff(muon_times)))
    return muon_times,muon_starts,muon_flechas

def rotate_x(element,angle,center=[0,0,0]): # rota un elemento (nos va a interesar el centro de los cuadrados del detector) en un ángulo alrededor del eje paralelo a x que contiene el punto center 
    sen = np.sin(angle)
    cos = np.cos(angle)

    x,y,z = (element-center).T # descentro para hacer la rotación alrededor de ese punto
    rotated = np.stack([x,cos*y-sen*z,sen*y+cos*z],axis=-1)+center # sumo de vuelta el centro para volver a la posición absoluta
    return rotated # calcula la matriz de rotación y se la aplica al vector (que tiene que ser un array) sin tener que pasar por el producto matricial

normal_angle = lambda ang : np.array([0.0,-np.sin(ang),np.cos(ang)]) # genera el vector normal al cuadrado de los detectores dado que rotaron un cierto ángulo desde la posición vertical, los signos son por como es una rotación alrededor del eje x

def ray_plane_intersection(start,flecha,center,normal): # calcula el punto en el que la traza de un muón intersecta el plano que contiene a algún cuadrado de los detectores
    denom = np.dot(normal,flecha)
    
    if np.abs(denom) < 1e-6: # evita casos límites que son complicados numéricamente
        return None
    
    t = np.dot(normal,center-start)/denom # la distancia recorrida desde el punto start hasta el de intersección
    return start+t*flecha # devuelve el punto en el que ocurre la intersección

def calc_uv(normal): # calcula unos vectores de proyecciones que sirven para ver si un punto está adentro del cuadrado
    u = np.cross(normal,np.array([0,0,1]))
    if np.linalg.norm(u) < 1e-6:
        u = np.cross(normal,np.array([0,1,0]))
    u = u/np.linalg.norm(u) # coordenada 1 en el plano del detector
    v = np.cross(normal,u)
    v = v/np.linalg.norm(v) # coordenada 2 en el plano del detector
    return u,v

def is_point_in_square(point,center,normal,u,v,half_len): # una vez que tengo las coordenadas en el plano y el punto de intersección, proyecto este punto sobre estas coordenadas y me fijo si está dentro del cuadrado de interés en el plano.
    local_point = point - center # calculo la posición relativa al centro del plano
    u_dist = np.dot(local_point,u) # proyección en la coordenada 1 del pkano
    v_dist = np.dot(local_point,v) # proyección en la coordenada 2 del plano

    epsilon = 1e-10 # para mirar los casos límites
    return (-half_len - epsilon <= u_dist <= half_len + epsilon) and (-half_len - epsilon <= v_dist <= half_len + epsilon)

def intersect_square(start,flecha,center,normal,u,v,half_len):
    point = ray_plane_intersection(start,flecha,center,normal)
    if point is None:
        return False
    return is_point_in_square(point,center,normal,u,v,half_len)

def intersect_detector(start,flecha,center_top,center_bottom,normal,u,v,half_len):
    top_intersection = intersect_square(start,flecha,center_top,normal,u,v,half_len)
    if top_intersection:
        return True
    else:
        bottom_intersection = intersect_square(start,flecha,center_bottom,normal,u,v,half_len)
    return bottom_intersection

def sim_rotados(lluvia,det_size,det_height,det_xy,det_delta,freq,ang0=None,seed=0):
    if ang0 is None: # si no doy una posición inicial definida sampleo de una uniforme
        rng = np.random.default_rng(seed) # defino el generador de números aleatorios
        ang0 = rng.uniform(0,2*np.pi) # sampleo el ángulo inicial en el que se encuentra el par de centelladores respecto al eje z

    # ahora voy a definir las posiciones (verticales, sin rotar) de las caras de cada centellador. Todas comparten el centro en el plano xy. La de abajo de todo está en z = 0, y el resto sube según la altura de los centelladores y la distancia entre los mismos.
    center_top = np.array([*det_xy,det_delta + det_height])
    center_top_up = center_top + np.array([0,0,det_height])
    center_bottom = center_top - np.array([0,0,det_delta])
    center_bottom_down = center_bottom - np.array([0,0,det_height])

    rot_center = np.array([*det_xy,det_height + det_delta/2]) # defino el punto alrededor del cual rotan los centelladores, que se va a usar para descentrar, tiene que ser el punto medio entre los centros en posición vertical.
    
    times,starts,flechas = lluvia
    detected = 0

    for t,start,flecha in zip(times,starts,flechas):
        ang = (2*np.pi*freq*t+ang0)%(2*np.pi) # actualizo la posición angular de los centelladores según la velocidad de rotación, normalizo a 2pi para usar menos memoria
        
        normal_new = normal_angle(ang) # calculo la nueva normal, la misma para todos los cuadrados
        u,v = calc_uv(normal_new) # calculo las proyecciones necesarias para las intersecciones

        # roto los centros del primer detector
        center_top_new = rotate_x(center_top,ang,center=rot_center)
        center_top_up_new = rotate_x(center_top_up,ang,center=rot_center)

        if intersect_detector(start,flecha,center_top_up_new,center_top_new,normal_new,u,v,det_size/2):
            # si pasó por el primer detector, roto los centros del segundo (no es necesario chequear los que ya no pasan por el primero porque no se cuentan como detecciones si no pasan por ambos, y es un cálculo que se repetiría muchas veces)
            center_bottom_new = rotate_x(center_bottom,ang,center=rot_center)
            center_bottom_down_new = rotate_x(center_bottom_down,ang,center=rot_center)
            if intersect_detector(start,flecha,center_bottom_new,center_bottom_down_new,normal_new,u,v,det_size/2):
                detected += 1
                # si pasaron por ambos detectores, registro el ángulo en el que fueron detectados

    return detected

if __name__ == '__main__':
    dist_centros = 24.7 # todas las distancias en cm
    det_height = 1.1 # altura de los centelladores
    det_size = 4.0 # ancho y largo de los centelladores
    det_delta = dist_centros - det_height # 23.6 # distancia entre la cara superior del centellador inferior y la cara inferior del centellador superior
    det_xy = [0,0] # punto en el plano xy que atraviesa el eje que contiene los centros de cada centellador en posición vertical
    margin = 0
    freq = 0 # vueltas por segundo

    detecteds = []
    for det_delta in np.arange(2,40+1):
        lluvia = generar_lluvia(det_size,det_delta,det_height,seed=0,n=int(1e5))
        detected = sim_rotados(lluvia,det_size,det_height,det_xy,det_delta,freq,ang0=0,seed=0)
        detecteds.append(detected)
    
    data = np.array(detecteds) # guardo la cantidad total de muones generados y los ángulos en los que se detectaron en un csv
    np.savetxt(f'data_distancias_plano.csv',data,delimiter=',')