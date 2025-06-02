import random
import numpy as np
from pycgp_finalclass.Genome import Genome
from pycgp_finalclass.Genome import CGPGenome
import matplotlib.pyplot as plt

class ES: #Evolution strategy
    def __init__(self, evaluator, mu, lam,genome_factory,mutation): 
        self.evaluator = evaluator
        self.mu = mu #parent population size
        self.lam = lam #offspring population size
        self.genome_factory= genome_factory #genome factory -> create genome function
        self.population = [self.genome_factory() for _ in range(mu)] 
        self.mutation = mutation
    
    #evolving process: evolve n time and stopping at a certain point without improvement
    def evolve(self, n_generations, early_stopping, verbose=False): #Put true in verbose to see prints
        parents = self.population
        best_genome = None
        best_fitness = -np.inf #start from the lowest value possible
        no_improvement = 0
        
        # List to track best fitness per generation
        fitness_history = []
        best_so_far = -np.inf  # best-so-far fitness
        
        evaluation_count = 0    

        for generation in range(n_generations):
            offspring = []
            for i in range(self.lam):
                parent = random.choice(parents)
                child = parent.copy()
                self.mutation.mutate(child)
                offspring.append(child)
                

            # Evaluate population
            population = parents + offspring
            scored_population = []
            for genome in population:
                fitness = self.evaluator.evaluate(genome,generation) # put the number of generations
                evaluation_count += 1
                if fitness > best_so_far:
                    best_so_far = fitness
                fitness_history.append((evaluation_count, best_so_far))  # ← append per evaluation
                scored_population.append((genome, fitness))

            # Sort the population based on fitness
            scored_population.sort(key=lambda x: x[1], reverse=True)

            # Show outputs of the top individuals
            if verbose:
                print("Top outputs this generation:")
                for i in range(min(5, len(scored_population))):
                    genome, fitness = scored_population[i]
                    print(f"  {i+1}: Output = {genome.outputs} fitness = {fitness:.4f}")


            # Update best genome if fitness improves
            if scored_population[0][1] > best_fitness:
                best_fitness = scored_population[0][1]
                best_genome = scored_population[0][0].copy()
                import copy
                best_genome = copy.deepcopy(scored_population[0][0]) #deepcopy to avoid mutating best_genome
                no_improvement = 0
            else:
                no_improvement += 1

            if no_improvement >= early_stopping:
                print(f"Early stopping at generation {generation} (no improvement for {early_stopping} generations).")
                break

            # Select the top individuals for the next generation, but reduce selection pressure
            parents = [genome for genome, _ in scored_population[:self.mu]]

            if verbose:
            # Print the top individual function string
                print(scored_population[0][0].to_function_string())
            

        print(f"\nBest fitness achieved: {best_fitness:.4f}")
        print(best_genome.to_function_string())
        self.plot_fitness_convergence(fitness_history)
        return best_genome
    
        
    def plot_fitness_convergence(self, fitness_history):
        evaluations, best_fitnesses = zip(*fitness_history)
        plt.figure(figsize=(10, 6))
        plt.plot(evaluations, best_fitnesses, marker='o', color='blue', linewidth=1)
        plt.title('Fitness Convergence Over Evaluations')
        plt.xlabel('Evaluation Count')
        plt.ylabel('Best-so-Far Fitness')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
