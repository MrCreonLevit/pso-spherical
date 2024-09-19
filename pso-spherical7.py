#%%
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import time

from numba import jit, njit

# Create a directory for saving plots if it doesn't exist
os.makedirs('plots', exist_ok=True)

# d = dimension of space (e.g. d=3 for spherical codes in 3D on "standard" sphere)
# n = number of d-dimensional points which make up the code (e.g. d=3, n=4 for tetrahedron)

#%%
#def pso(f, d, population_size=1000, max_iterations=100, w=0.99, c1=0.2, c2=0.2, c3=0.01,verbose=True):
#def pso(f, d, population_size=1000, max_iterations=100, w=0.5, c1=0.5, c2=0.5, c3=1.0,verbose=True):
#def pso(f, d, population_size=1000, max_iterations=100, w=0.1, c1=0.2, c2=0.3, c3=1.0, verbose=True):
#def pso(f, d, population_size=1000, max_iterations=100, w=0.5, c1=0.2, c2=0.3, c3=0.1, verbose=True):
#def pso(f, d, population_size=1000, max_iterations=100, w=0.5, c1=0.5, c2=0.5, c3=0.5, verbose=True):
#def pso(f, d, population_size=1000, max_iterations=100, w=0.5, c1=0.3, c2=0.2, c3=0.1, verbose=True):
#def pso(f, d, population_size=1000, max_iterations=100, w=0.1, c1=0.3, c2=0.5, c3=0.1, verbose=True):
# def pso(f, d, population_size=1000, max_iterations=100, w=0.1, c1=0.3, c2=0.5, c3=0.1, verbose=True):
# def pso(f, d, n, population_size=1000, max_iterations=100, w=0.5, c1=0.3, c2=0.5, c3=0.1, verbose=True):
# def pso(f, d, n, population_size=1000, max_iterations=100, w=0.95, c1=0.5, c2=0.4, c3=0.2, verbose=True): # 24-cell
# def pso(f, d, n, population_size=1000, max_iterations=100, w=0.95, c1=0.5, c2=0.1, c3=0.2, verbose=True): 
# def pso(f, d, n, population_size=1000, max_iterations=100, w=0.95, c1=0.5, c2=0.4, c3=0.2, verbose=True, plots=True): #workhorse
def pso(f, d, n, population_size=1000, max_iterations=100, w=0.95, c1=0.1, c2=0.4, c3=0.5, verbose=True, plots=False): 
    # log parameters
    print(f"PSO parameters: d={d}, n={n}, population_size={population_size}, max_iterations={max_iterations}, w={w}, c1={c1}, c2={c2}, c3={c3}")

    total_fitness_evals = 0
    
    # Initialize particles and velocities and fitness
    particles = np.random.normal(0, 1, (population_size, d, n))
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
        
        ####
        # OK all of these are slower than the default serial fitness calculation
        # because the overhead of the parallel environment is just too high.
        # and the time spent in the fitness function for each particle is so small (~15us)

        #why is this slower?
        # from multiprocessing import Pool
        # @jit(nopython=True)
        #@njit(nogil=True)
        #def calculate_fitness(p):
        #    return f(p)
        # with Pool(processes=10) as pool:
        #     fitness = np.array(pool.map(calculate_fitness, particles))
        # than the alternative
        
        #why is this slower?
        # from numba import jit, prange
        # @jit(nopython=True, parallel=True)
        # def parallel_f(particles):
        #     result = np.empty(len(particles))
        #     for i in prange(len(particles)):
        #         result[i] = f(particles[i])
        #     return result

         #why is this slower?
        # from joblib import Parallel, delayed
        # fitness = np.array(Parallel(n_jobs=10, backend='multiprocessing')(delayed(f)(p) for p in particles))

        #why is this slower?
        # from concurrent.futures import ProcessPoolExecutor
        # if __name__ == '__main__':
        #     with ProcessPoolExecutor(max_workers=10) as executor:
        #         fitness = np.array(list(executor.map(f, particles)))

        # default serial fitness calculation
        fitness = np.array([f(p) for p in particles])
        total_fitness_evals += population_size
        # print(f"iteration {t}")

        # Calculate and store average fitness
        average_fitness = np.mean(fitness)
        if plots:
            average_fitness_history.append(average_fitness)
        # print(f"Average fitness: {average_fitness}")

        # Update personal best
        improved = fitness > personal_best_fitness
        personal_best[improved] = particles[improved]
        personal_best_fitness[improved] = fitness[improved]
        
        # Update global best
        if np.max(fitness) > global_best_fitness:
            best_index = np.argmax(fitness)
            global_best = particles[best_index].copy()
            global_best_fitness = fitness[best_index]
            if verbose:
                print(f"Iteration {t:_}: New global best fitness: {global_best_fitness} = {math.degrees(global_best_fitness):.2f} deg.")
                # print(f"  Position: {global_best}")
                # print(f"  Std: {np.std(particles, axis=0)}")
                print(f"total fitness evals: {total_fitness_evals:_}")
                print(f"avg. stddev: {np.mean(np.std(particles, axis=0))}")
       
        # Check for significant change every 10% of iterations
        if (t + 1) % check_interval == 0:

            relative_change = abs(global_best_fitness - old_global_best_fitness) / max(abs(old_global_best_fitness), 1e-10)
            if relative_change > 1e-6:
                print(f"Iteration {t+1:_}: Global best fitness has changed significantly.")
                print(f"  total fitness evals: {total_fitness_evals:_}")
                print(f"  Old fitness: {old_global_best_fitness} = {math.degrees(old_global_best_fitness):.2f} deg.")
                print(f"  New fitness: {global_best_fitness} = {math.degrees(global_best_fitness):.2f} deg.")
            else:
                print(f"Iteration {t+1:_}: Global best fitness has not changed significantly.")
                max_iterations = t
                break # could keep going
            old_global_best_fitness = global_best_fitness
          
        
        # Store global best for plotting
        if plots:
            global_best_history.append(global_best.copy())
            fitness_history.append(global_best_fitness)

    
    final_variance = np.var(particles, axis=0)
    
    if plots:
        # Convert lists to numpy arrays for easier plotting
        mean_history = np.array(mean_history)
        variance_history = np.array(variance_history)
        std_history = np.array(std_history)
        # Plot best and average fitness over iterations
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(fitness_history)), fitness_history, label='Best Fitness')
        plt.plot(range(len(average_fitness_history)), average_fitness_history, label='Average Fitness')
        plt.title('PSO: Best and Average Fitness Over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Fitness')
        plt.legend()
        plt.grid(True)
        plt.savefig('plots/pso_fitness_comparison.png')
        plt.close()
    
        # New plot for mean of particles
        plt.figure(figsize=(10, 5))
        for i in range(d):
            plt.plot(range(len(mean_history)), mean_history[:, i], label=f'Dimension {i+1}')
        plt.title('PSO: Mean of Particles Over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Mean Value')
        plt.legend()
        plt.grid(True)
        plt.savefig('plots/pso_particle_mean.png')
        plt.close()

        # Plot global best over iterations
        plt.figure(figsize=(10, 5))
        global_best_history = np.array(global_best_history)
        for i in range(d):
            plt.plot(range(len(global_best_history)), global_best_history[:, i], label=f'Dimension {i+1}')
        plt.title('PSO: Global Best Position Over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Position Value')
        plt.legend()
        plt.grid(True)
        plt.savefig('plots/pso_global_best.png')
        plt.close()

        # New plot for variance of particles
        plt.figure(figsize=(10, 5))
        for i in range(d):
            plt.plot(range(len(variance_history)), variance_history[:, i], label=f'Dimension {i+1}')
        plt.title('PSO: Variance of Particles Over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Variance Value')
        plt.legend()
        plt.grid(True)
        plt.savefig('plots/pso_particle_variance.png')
        plt.close()

        # New plot for standard deviation of particles
        plt.figure(figsize=(10, 5))
        for i in range(d):
            plt.plot(range(len(std_history)), std_history[:, i], label=f'Dimension {i+1}')
        plt.title('PSO: Standard Deviation of Particles Over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Standard Deviation Value')  
        plt.legend()
        plt.grid(True)
        plt.savefig('plots/pso_particle_std.png')
        plt.close()
    
    # After the main PSO loop, add this gradient descent fine-tuning step
    initial_learning_rate = 0.001
    max_gd_iterations = 1_000_000
    gd_epsilon = 1e-11

    print("-------")
    print(f"Starting gradient descent fine-tuning with initial learning rate {initial_learning_rate}...")
    for i in range(max_gd_iterations):
        # Decay the learning rate
        learning_rate = initial_learning_rate / (1 + i / 100_000)  # You can adjust the decay rate

        old_fitness = f(global_best)
        gradient = calculate_gradient(f, global_best)
        
        # Update global_best using gradient
        global_best += learning_rate * gradient
        
        # Normalize the vectors
        global_best /= np.linalg.norm(global_best, axis=0, keepdims=True)
        
        new_fitness = f(global_best)
        total_fitness_evals += 1
        
        if (i + 1) % (max_gd_iterations/1000) == 0:
            print(f"GD Iteration {i+1:_}: fitness = {math.degrees(new_fitness):.4f} deg")
            print(f"total fitness evals: {total_fitness_evals:_}, ", end='')
            print(f"Learning rate: {learning_rate:.6f}")
        
        # Check for convergence
        if abs(new_fitness - old_fitness) < gd_epsilon:
            if verbose:
                print(f"Gradient descent converged after {i+1:_} iterations")
            break
    
    final_fitness = f(global_best)
    if verbose:
        print(f"Final fitness after gradient descent: {math.degrees(final_fitness):.4f} deg")
        print(f"Total fitness evaluations: {total_fitness_evals:_}")

    return global_best, final_variance


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
    angles = np.arccos(dots)
    # print(f"dots: {dots}")
    # print(f"angles: {angles}")
    # print(f"min angle: {np.min(np.ravel(angles))}")
    return np.min(np.ravel(angles))

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

def calculate_population_gradients(f, population, epsilon=1e-8):
    return np.array([calculate_gradient(f, p, epsilon) for p in population])

#%%
# Optimize test functions
if __name__ == "__main__":
    # Set a fixed seed for reproducibility. You can use any integer value as the seed
    np.random.seed(int(time.time()))
    print("Optimizing minimum angle with PSO:")
    # result, final_variance = pso(min_angle, d=4, n=12, population_size=50, max_iterations=200_000) #works (24-cell)
    # result, final_variance = pso(min_angle, d=4, n=24, population_size=30, max_iterations=10_000_000, plots=False) # almost finds E8?
    result, final_variance = pso(min_angle, d=5, n=24, population_size=30, max_iterations=1_000_000, plots=False)
    
    # print(f"Optimum found at: {result}")
    print(f"optimum fitness: {math.degrees(min_angle(result)):.4f} deg.")
    
    # Calculate gradient for the best solution
    
    # best_solution_gradient = calculate_gradient(min_angle, result)
    # print(f"Gradient for the best solution:")
    # print(best_solution_gradient)

# %%
