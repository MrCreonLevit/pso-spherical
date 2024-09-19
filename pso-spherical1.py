#%%
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import time

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
def pso(f, d, n, population_size=1000, max_iterations=100, w=0.95, c1=0.5, c2=0.4, c3=0.2, verbose=True): # 24-cell
    limit = 10.0
    # Initialize particles and velocities and fitness
    particles = np.random.normal(0, 1, (population_size, d, n))
    particles = particles / np.linalg.norm(particles, axis=1, keepdims=True)
    velocities = np.random.normal(0, 0.1, (population_size, d, n))
    fitness = np.array([f(p) for p in particles])
    
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
                      )) * (1 - (t/max_iterations)) # final divisor term is "cooling" term.
        particles += velocities
        # particles = np.clip(particles, -limit, limit)
        particles = particles / np.linalg.norm(particles, axis=1, keepdims=True)
        
        # Evaluate fitness
        fitness = np.array([f(p) for p in particles])
        
        # Calculate and store average fitness
        average_fitness = np.mean(fitness)
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
                print(f"Iteration {t}: New global best fitness: {global_best_fitness} = {math.degrees(global_best_fitness):.2f} deg.")
                # print(f"  Position: {global_best}")
                # print(f"  Std: {np.std(particles, axis=0)}")
                print(f"  avg. stddev: {np.mean(np.std(particles, axis=0))}")
                # Check for significant change every 10% of iterations
        if (t + 1) % check_interval == 0:
            relative_change = abs(global_best_fitness - old_global_best_fitness) / max(abs(old_global_best_fitness), 1e-10)
            if relative_change > 1e-6:
                print(f"Iteration {t+1}: Global best fitness has changed significantly.")
                print(f"  Old fitness: {old_global_best_fitness} = {math.degrees(old_global_best_fitness):.2f} deg.")
                print(f"  New fitness: {global_best_fitness} = {math.degrees(global_best_fitness):.2f} deg.")
            else:
                print(f"Iteration {t+1}: Global best fitness has not changed significantly.")
                max_iterations = t
                break # could keep going
            old_global_best_fitness = global_best_fitness
        
        # Store global best for plotting
        global_best_history.append(global_best.copy())
        fitness_history.append(global_best_fitness)

    
    final_variance = np.var(particles, axis=0)
    
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
    
    return global_best, final_variance


def ripple_function_N_d(x):
    r = np.sqrt(np.sum( x**2))
    return np.cos(r) / (1 + r)

def hard1_function_N_d(x):
    return ripple_function_N_d(x - 10) + 0.9 * ripple_function_N_d(x + 10)

# we want to maximise the minimum angle between any two points
def min_angle(x):
    dots = np.dot(x.T, x)
    np.fill_diagonal(dots, -1)
    angles = np.arccos(dots)
    # print(f"dots: {dots}")
    # print(f"angles: {angles}")
    # print(f"min angle: {np.min(np.ravel(angles))}")
    return np.min(np.ravel(angles))

#%%
# Optimize test functions
if __name__ == "__main__":
    # Set a fixed seed for reproducibility. You can use any integer value as the seed
    np.random.seed(int(time.time()))
    print("Optimizing minimum angle with PSO:")
    # result, final_variance = pso(min_angle, d=4, n=12, population_size=50, max_iterations=200000) #works (24-cell)
    result, final_variance = pso(min_angle, d=3, n=9, population_size=50, max_iterations=200000) 
    # print(f"Optimum found at: {result}")
    print(f"optimum fitness: {math.degrees(min_angle(result)):.2f} deg.")
    # print(f"Final particle variance: {final_variance}")

    
# %%
