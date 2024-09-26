import os
import time
import sys
from datetime import datetime
from time import sleep
import numpy as np
import math

from numba import jit, njit
import dask

def cool(x, k=5.0):  # k=3 for a relatively steep curve
    if x >= 1:
        return 1  # To ensure it approaches 1 as x approaches or equals 1
    return (2 / (1 + math.exp(-k * (x / (1 - x)))))-1

# try @jitting the min_angle function with @jit decorator
#@jit(nopython=True,)
@njit(nogil=True)
def min_angle(x):
    dots = np.dot(x.T, x)
    # set diagonal to -1 so that we don't count self-dots (dot==1, angle==0)
    np.fill_diagonal(dots, -1)
    np.clip(dots, -1, 1, out=dots)
    angles = np.arccos(dots)
    # print(f"dots: {dots}")
    # print(f"angles: {angles}")
    # print(f"min angle: {np.min(np.ravel(angles))}")
    return np.min(np.ravel(angles))

#@njit(nogil=True)
def avg_angle(x):
    dots = np.dot(x.T, x)

    mask = np.triu(np.ones(dots.shape, dtype=bool), k=1)
    
    # Use the mask to select the upper triangular elements
    upper_tri_values = dots[mask]
    angles = np.arccos(upper_tri_values)

    return np.mean(angles)

#@jit(nopython=True,)
@njit(nogil=True)
def calculate_gradient(f, x, epsilon=1e-6):
    d, n = x.shape
    gradient = np.zeros_like(x)
    
    for i in range(d):
        for j in range(n):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i, j] += epsilon
            x_minus[i, j] -= epsilon
            
            # Normalize the perturbed vectors
            x_plus[:, j] /= np.linalg.norm(x_plus[:, j])
            x_minus[:, j] /= np.linalg.norm(x_minus[:, j])
            
            gradient[i, j] = (f(x_plus) - f(x_minus)) / (2 * epsilon)
    
    return gradient


# Create a directory for saving plots if it doesn't exist
os.makedirs('plots', exist_ok=True)

# d = dimension of space (e.g. d=3 for spherical codes in 3D on "standard" sphere)
# n = number of d-dimensional points which make up the code (e.g. d=3, n=4 for tetrahedron)

@dask.delayed
def pso(f, d, n, population_size=1000, max_iterations=100, w=0.95, c1=0.5, c2=0.0, c3=0.2, verbose=False, plots=True, thread=0): #workhorse
    # log parameters
    # Generate a unique filename based on the current timestamp
    output_filename = f"pso_output_{thread}.log"
    # Open the file for writing
    output_file = open(output_filename, 'w')    
    print(f"PSO parameters: d={d}, n={n}, population_size={population_size}, max_iterations={max_iterations}, w={w}, c1={c1}, c2={c2}, c3={c3}", 
          file=output_file, flush=True)

    total_fitness_evals = 0
    
    # Initialize particles and velocities and fitness
    particles = np.random.normal(0, 1, (population_size, d, n))
    # try antipodal initialization of points if there are an even number of points
    # why does this break things?
    if n % 2 == 0:
        for i in range(n//2):
            particles[:,:,n//2+i] = -np.copy(particles[:,:,i])
   
    particles = particles / np.linalg.norm(particles, axis=1, keepdims=True)
    velocities = np.random.normal(0, 0.1, (population_size, d, n))
    fitness = np.array([f(p) for p in particles])  # should parallelize this
    
    # Initialize personal best and global best
    personal_best = particles.copy()
    personal_best_fitness = np.array([f(p) for p in personal_best])
    global_best = personal_best[np.argmax(personal_best_fitness)]
    global_best_fitness = np.max(personal_best_fitness)
    old_global_best_fitness = global_best_fitness
    
    # Lists to store history for plotting
    global_best_history = [global_best.copy()]
    fitness_history = [global_best_fitness]
    average_fitness_history = []
    mean_history = []
    variance_history = []
    std_history = []

    check_interval = max(1, max_iterations // 10)  # Check every 10% of iterations

    for t in range(max_iterations):
        # Calculate and store mean and variance of particles
        particle_mean = np.mean(particles, axis=0)
        particle_variance = np.var(particles, axis=0)
        particle_std = np.std(particles, axis=0)
        if plots:
            mean_history.append(particle_mean)
            variance_history.append(particle_variance)
            std_history.append(particle_std)
        
        # Update velocities and positions
        r1, r2 = np.random.rand(2)
        velocities = (w * velocities + 
                      (1-w)*(
                        c1 * r1 * (personal_best - particles) + 
                        c2 * r2 * (global_best - particles) +
                        c3 * np.random.normal(0, 1, (population_size, d, n))
                    #   )) * (1 - (t/max_iterations)**(0.25)) # final quotient term is "cooling" term.
                      )) * (1.0 - cool(t/max_iterations))
        particles += velocities
        
        particles = particles / np.linalg.norm(particles, axis=1, keepdims=True)
        
        # if (t + 1) % 1000 == 0:
        #     # for each particle, take a step in the direction of its gradient
        #     # print(f"Iteration {t+1}: Taking a step in the direction of the gradient.")
        #     print('.', end='', flush=True)
        #     population_gradients = calculate_population_gradients(f, particles)
        #     particles += 0.001*(population_gradients)
        #     particles = particles / np.linalg.norm(particles, axis=1, keepdims=True)

        # default serial fitness calculation
        fitness = np.array([f(p) for p in particles])
        total_fitness_evals += population_size  # this is the number of fitness evaluations
        # print(f"iteration {t}, fitness: {fitness}")

        # Calculate and store average fitness
        average_fitness = np.mean(fitness)
        if plots:
            average_fitness_history.append(average_fitness)
        # print(f"Average fitness: {average_fitness}")

        # Update personal best
        improved = fitness > personal_best_fitness
        personal_best[improved] = particles[improved]
        personal_best_fitness[improved] = fitness[improved]
        
        # print(f"Iteration {t:_}: New global best fitness: {f(global_best)} = {math.degrees(f(global_best)):.2f} deg.")
        # print(f"Iteration {t:_}: New min angle: {min_angle(global_best)} = {math.degrees(min_angle(global_best)):.2f} deg.")
                
        # Update global best
        if np.max(fitness) > global_best_fitness:
            best_index = np.argmax(fitness)
            global_best = particles[best_index].copy()
            global_best_fitness = fitness[best_index]
            if verbose or True:
                print(f"Iteration {t:_}: New global best fitness: {f(global_best)} = {math.degrees(f(global_best)):.10f} deg.", 
                      file=output_file, flush=True)
                # print(f"Iteration {t:_}: New global best avg angle: {avg_angle(global_best)} = {math.degrees(avg_angle(global_best)):.2f} deg.")      
                # print(f"  Position: {global_best}")
                # print(f"  Std: {np.std(particles, axis=0)}")
                print(f"total fitness evals: {total_fitness_evals:_}, ", end='', file=output_file, flush=True)
                print(f"avg. stddev: {np.mean(np.std(particles, axis=0)):.5f}", file=output_file, flush=True)
       
        # Check for significant change every 10% of iterations
        if (t + 1) % check_interval == 0:

            relative_change = abs(global_best_fitness - old_global_best_fitness) / max(abs(old_global_best_fitness), 1e-10)
            if relative_change > 1e-7:
                print(f"Iteration {t+1:_}: Global best fitness has changed significantly.", file=output_file, flush=True)
                print(f"  total fitness evals: {total_fitness_evals:_}", file=output_file, flush=True)
                print(f"  Old fitness: {old_global_best_fitness} = {math.degrees(old_global_best_fitness):.2f} deg.", file=output_file, flush=True)
                print(f"  New fitness: {global_best_fitness} = {math.degrees(global_best_fitness):.2f} deg.", file=output_file, flush=True)
            else:
                print(f"Iteration {t+1:_}: Global best fitness has not changed significantly.", file=output_file, flush=True)
                max_iterations = t
                break # could keep going
            old_global_best_fitness = global_best_fitness
          
        
        # Store global best for plotting
        if plots:
            global_best_history.append(global_best.copy())
            fitness_history.append(global_best_fitness)
 
    final_variance = np.var(particles, axis=0)
    
    # After the main PSO loop, add this Adam optimization step
    print("-------", file=output_file, flush=True)
    print("Starting Adam optimization...", file=output_file, flush=True)

    # Adam parameters
    alpha = 0.001  # learning rate
    beta1 = 0.9  # exponential decay rate for first moment estimates
    beta2 = 0.999  # exponential decay rate for second moment estimates
    epsilon = 1e-8  # small constant to prevent division by zero
    max_adam_iterations = 50_000
    adam_epsilon = 1e-11  # convergence criterion
    true_best_fitness = 0.0

    # Initialize Adam variables
    m = np.zeros_like(global_best)
    v = np.zeros_like(global_best)
    t = 0

    for i in range(max_adam_iterations):
        t += 1
        old_fitness = f(global_best)
        gradient = calculate_gradient(f, global_best)
        
        # Update biased first moment estimate
        m = beta1 * m + (1 - beta1) * gradient
        # Update biased second raw moment estimate
        v = beta2 * v + (1 - beta2) * np.square(gradient)
        
        # Compute bias-corrected first moment estimate
        m_hat = m / (1 - beta1**t)
        # Compute bias-corrected second raw moment estimate
        v_hat = v / (1 - beta2**t)
        
        # Update global_best
        global_best += alpha * m_hat / (np.sqrt(v_hat) + epsilon)
        
        # Normalize the vectors
        global_best /= np.linalg.norm(global_best, axis=0, keepdims=True)
        
        new_fitness = f(global_best)
        # print(f"Adam Iteration {i+1:_}: fitness = {math.degrees(new_fitness):.4f} deg")
        total_fitness_evals += 1
        if new_fitness > true_best_fitness:
            true_best_fitness = new_fitness
            print(f"Adam Iteration {i+1:_}: fitness = {math.degrees(true_best_fitness):.4f} deg", file=output_file, flush=True)

        
        if verbose and (i + 1) % (max_adam_iterations//100) == 0:
            print(f"Adam Iteration {i+1:_}: fitness = {math.degrees(new_fitness):.4f} deg", file=output_file, flush=True)
            print(f"total fitness evals: {total_fitness_evals:_}", file=output_file, flush=True)
        
        # Check for convergence (should we compare to true_best_fitness?)
        if abs(new_fitness - old_fitness) < adam_epsilon:
            print(f"Adam optimization converged after {i+1:_} iterations", file=output_file, flush=True)
            break

    final_fitness = f(global_best)
    print(f"Final fitness after Adam optimization: {math.degrees(final_fitness):.4f} deg", file=output_file, flush=True)
    print(f"Total fitness evaluations: {total_fitness_evals:_}", file=output_file, flush=True)
    print(f"True best fitness found: {math.degrees(true_best_fitness):.4f} deg", file=output_file, flush=True)
    return math.degrees(true_best_fitness)


@dask.delayed
def work(i):
    sleep(5)
    print(f"Finished work {i}", flush=True)
    return i*i


# Set up Dask client
# client = Client()

# Number of PSO runs
num_runs = 10


# Optimize test functions
if __name__ == "__main__":
    # Set a fixed seed for reproducibility. You can use any integer value as the seed
    np.random.seed(int(time.time()))
    print("Optimizing minimum angle with PSO:",flush=True)
    results = []
    for i in range(num_runs):
        print(f"Queing up PSO {i+1} of {num_runs}",flush=True)
        y = dask.delayed(pso)(min_angle, 4, 12, population_size=30, max_iterations=100, c2=0.0, plots=False, thread=i)
        #y = dask.delayed(work)(i)
        results.append(y)
    all_results = dask.compute(*results)    
    print(all_results)    