import numpy as np

def sample_theta_cos2(rng, N):
    theta_vals = []
    while len(theta_vals) < N:
        size = 2 * (N - len(theta_vals))  # Oversample to improve efficiency
        theta = rng.uniform(0, np.pi/2, size)
        accept = rng.uniform(0, 1, size) < np.cos(theta)**2
        theta_vals.extend(theta[accept])
    
    return np.array(theta_vals[:N])

if __name__ == '__main__':
    data = sample_theta_2_cos2(np.random.default_rng(seed=0),int(1e7))
    
    np.savetxt(f'data_2_rot_lluvia.csv',data,delimiter=',')