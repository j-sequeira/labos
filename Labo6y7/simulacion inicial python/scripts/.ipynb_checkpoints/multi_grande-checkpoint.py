import numpy as np
from multiprocessing import Pool

def batch_muon_directions(rng,N): # genera la dirección en la que se mueven N muones
    directions = np.empty((N,3)) # inicializa el array a llenar
    filled = 0
    while filled < N: # rejection sampling para el cos^2 con theta
        size = 2*(N-filled) # hace siempre el doble porque ese cómputo es menos costoso que repetir el loop
        theta = rng.uniform(0,np.pi/2,size)
        accept = rng.uniform(0,1,size) < np.cos(theta)**2 # chequea todos de una, más rápido
        theta = theta[accept]

        phi = rng.uniform(0,2*np.pi,len(theta)) # samplea phi de una uniforme
        # estas dos líneas son si se quiere hacer 2D y que salga desde los dos sentidos y no de un solo lado
#        signo = rng.integers(0,2,len(theta))
#        phi = np.pi/2 + signo*np.pi
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

def batch_generate_muon_t(det_size,det_delta,det_height,margin,z_margin,rate,rng,N):
    x_length = det_size # la longitud del plano en la dirección en la que no roto es el lado de los centelladores
    y_length = det_delta+2*det_height+2*margin # en la dirección en la que roto es la longitud ocupada por los centelladores (distancia + alturas) sumando un margen a elección
    area = x_length*y_length
    
    ts = rng.exponential(scale=1/(rate*area),size=N) # sampleo el tiempo entre muones de una exponencial
    ts = np.cumsum(ts) # como samplear de la exponencial me da los interavlos entre muones, los sumo para tener los tiempos absolutos
    
    x_o = rng.uniform(-x_length/2,x_length/2,size=N) # sampleo de una uniforme 2D en el plano de generación
    y_o = rng.uniform(-y_length/2,y_length/2,size=N)
#    x_o = np.zeros(N) # si quiero considerar un único punto de origen en el centro del plano
#    y_o = np.zeros(N)
    z0 = det_height+det_delta/2
    z_length = det_delta+2*det_height+2*z_margin
    z_o = rng.uniform(z0-z_length/2,z0+z_length/2,size=N) # se repite el mismo plano a lo largo de varias alturas de generación, se samplea desde una uniforme para obtener ese valor
#    z_o = np.full(N,det_height+det_delta/2) # la altura de generación es el punto medio entre los centelladores
    starts = np.stack([x_o,y_o,z_o],axis=1) # junto todo en un array de N vectores de 3 componentes

    flechas = batch_muon_directions(rng,N)
    return ts,starts,flechas

def batch_generate_muon_t_until(det_size,det_delta,det_height,margin,rate,rng,T,multiplier=1.2):
    x_length = det_size # la longitud del plano en la dirección en la que no roto es el lado de los centelladores
    y_length = det_delta+2*det_height+2*margin # en la dirección en la que roto es la longitud ocupada por los centelladores (distancia + alturas) sumando un margen a elección
    area = x_length*y_length

    ts = np.array([0,1])
    while np.sum(ts) < T:
        ts = rng.exponential(scale=1/(rate*area),size=int(T/(rate*area)*multiplier)) # sampleo el tiempo entre muones de una exponencial acá para lograr que el tiempo final sea mayor o igual al pedido T, sampleo de la exponencial hasta que esto pase. El sampleo requiere un N fijo, así que le doy la proporción entre el T pedido y el tiempo medio de generación (que es la esperanza de la cantidad de muones generados en ese tiempo) multiplicado por un valor a elección para tener un colchón extra, y todo redondeado al entero más cercano

    ts = np.cumsum(ts) # como samplear de la exponencial me da los interavlos entre muones, los sumo para tener los tiempos absolutos
    N = len(ts)
    
    x_o = rng.uniform(-x_length/2,x_length/2,size=N) # sampleo de una uniforme 2D en el plano de generación
    y_o = rng.uniform(-y_length/2,y_length/2,size=N)
    z_o = np.full(N,det_height+det_delta/2) # la altura de generación es el punto medio entre los centelladores
    starts = np.stack([x_o,y_o,z_o],axis=1) # junto todo en un array de N vectores de 3 componentes

    flechas = batch_muon_directions(rng,N)
    return ts,starts,flechas

def generar_lluvia(mode,det_size,det_delta,det_height,margin,z_margin,seed=0,rate=1/60,t_f=0,n=0,multiplier=1.2):
    rng = np.random.default_rng(seed) # defino el generador de números aleatorios

    if mode == 'n':  # si quiero un número de muones fijo
        muon_times,muon_starts,muon_flechas = batch_generate_muon_t(det_size,det_delta,det_height,margin,z_margin,rate,rng,n)

    elif mode == 't':  # si quiero un tiempo de simulación fijo
        muon_times,muon_starts,muon_flechas = batch_generate_muon_t_until(det_size,det_delta,det_height,margin,rate,rng,t_f,multiplier=multiplier)

    else:
        raise ValueError("Modo incorrecto: use 'n' (número) o 't' (tiempo)")
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
    
    measured_angles = []
    times,starts,flechas = lluvia

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
                measured_angles.append(ang)
                # si pasaron por ambos detectores, registro el ángulo en el que fueron detectados

    return measured_angles,len(times)

def lluvia_chunker(lluvia,n_chunks): # divido la lluvia en n_chunks pedazos para correr partes separadas en paralelo
    times,starts,flechas = lluvia
    times_arrays = np.array_split(times,n_chunks) # esta función divide el array en pedazos de aproximadamente el mismo tamaño
    starts_arrays = np.array_split(starts,n_chunks)
    flechas_arrays = np.array_split(flechas,n_chunks)
    return list(zip(times_arrays, starts_arrays, flechas_arrays))

def sim_rotados_single(args):
    lluvia_chunk,det_size,det_height,det_xy,det_delta,freq,ang0,seed = args
    return sim_rotados(lluvia_chunk,det_size,det_height,det_xy,det_delta,freq,ang0=ang0,seed=seed)

def sim_parallel(lluvia,det_size,det_height,det_xy,det_delta,freq,n_chunks,ang0=None,seed=0):
    chunks = lluvia_chunker(lluvia,n_chunks)

    if ang0 is None:
        rng = np.random.default_rng(seed)
        ang0 = rng.uniform(0,2*np.pi)
        
    # junto todos los argumentos de la función sim_rotados en 1 para poder correr en paralelo
    args_list = [
        (chunk,det_size,det_height,det_xy,det_delta,freq,ang0,seed)
        for i,chunk in enumerate(chunks)
    ]

    with Pool(processes=n_chunks) as pool:
        results = pool.map(sim_rotados_single,args_list)

    total_angles = []
    total_muons = 0
    for angles,count in results:
        total_angles.extend(angles)
        total_muons += count

    return total_angles,total_muons

if __name__ == '__main__':
    dist_centros = 24.7 # todas las distancias en cm
    det_height = 1.1 # altura de los centelladores
    det_size = 4.0 # ancho y largo de los centelladores
    det_delta = dist_centros - det_height # 23.6 # distancia entre la cara superior del centellador inferior y la cara inferior del centellador superior
    det_xy = [0,0] # punto en el plano xy que atraviesa el eje que contiene los centros de cada centellador en posición vertical
    multiplier = 10
    margin = (multiplier-1)*(det_delta+2*det_height)/2 # distancia a la que se amplia a cada lado el plano de generación
    z_margin = (multiplier-1)*(det_delta+2*det_height)/2 # distancia la que se amplia hacia arriba y a bajo el volumen de generación
    freq = 1 # vueltas por segundo
    n_chunks = 4 # cuántos procesos en paralelo quiero correr
    
    lluvia = generar_lluvia('n',det_size,det_delta,det_height,margin,z_margin,seed=0,n=int(1e7))
    measured_angles,total_muons = sim_rotados(lluvia,det_size,det_height,det_xy,det_delta,freq,ang0=None,seed=0)
    
    data = np.array([total_muons,*measured_angles]) # guardo la cantidad total de muones generados y los ángulos en los que se detectaron en un csv
    np.savetxt('data_2_rot_grande.csv',data,delimiter=',')

