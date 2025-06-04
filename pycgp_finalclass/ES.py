import random
import numpy as np
from pycgp_finalclass.Genome import Genome
from pycgp_finalclass.Genome import CGPGenome
import matplotlib.pyplot as plt
import copy
class ES: #Evolution strategy
    def __init__(self, evaluator, lam,parent,mutation): 
        self.evaluator = evaluator
        self.lam = lam #offspring population size
        self.parent = parent
        self.mutation = mutation
    
    #evolving process: evolve n time and stopping at a certain point without improvement
    def evolve(self, n_generations, early_stopping, verbose=False): #Put true in verbose to see prints
        parent = self.parent
        best_genome = self.parent #deepcopy to avoid mutating best_genome
        best_fitness = self.evaluator.evaluate(parent) #start from the lowest value possible
        print(f"Starting fitness {best_fitness:.4f}")
        no_improvement = 0
        
        # List to track best fitness per generation
        fitness_history = []
        evaluation_count = 0    

        for generation in range(n_generations):
            if verbose:
                print(f"Generation {generation}")
            offspring = []
            for i in range(self.lam):
                parent = best_genome.copy() #deepcopy to avoid mutating best_genome
                child = parent.copy()
                self.mutation.mutate(child)
                offspring.append(child)
                

            # Evaluate population
            scored_population = []
            for genome in offspring:
                #dont need to evalua
                fitness = self.evaluator.evaluate(genome) # put the number of generations
                scored_population.append((genome, fitness))
                # Sort the population based on fitness
                scored_population.sort(key=lambda x: x[1], reverse=True)
                #for fitness plotting
                evaluation_count += 1
                fitness_history.append((evaluation_count, best_fitness))

            # Update best genome if fitness improves
            if scored_population[0][1] > best_fitness:
                best_fitness = scored_population[0][1]
                best_genome = scored_population[0][0].copy() #deepcopy to avoid mutating best_genome
                no_improvement = 0
            else:
                no_improvement += 1

            if no_improvement >= early_stopping:
                print(f"Early stopping at generation {generation} (no improvement for {early_stopping} generations).")
                break


            
            # Show outputs of the top individuals
            '''if verbose:
                print(f"Best fitness this generation: {scored_population[0][1]:.4f}")
                print("Top outputs this generation:")
                for i in range(min(5, len(scored_population))):
                    genome, fitness = scored_population[i]
                    print(f"  {i+1}: Output = {genome.outputs} fitness = {fitness:.4f}")'''


            if verbose:
            # Print the top individual function string
                print(f"\nBest fitness this generation: {best_fitness:.4f}")
                print(best_genome.to_function_string())
            

        print(f"\nBest fitness achieved: {best_fitness:.4f}")
        print(best_genome.to_function_string())
        self.plot_fitness_convergence(fitness_history)
        best_genome.visualize_active_graph()
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
