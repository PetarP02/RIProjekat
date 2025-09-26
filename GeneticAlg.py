from genome import Genome
from node import Node
import numpy as np
import copy as cp
import random
import time
from typing import Union


class GenProg:
    """
    Implements a genetic programming algorithm to evolve expressions approximating a target function.

    Each genome in the population is represented as a binary expression tree. The algorithm
    iteratively evolves the population using selection, crossover, and mutation, tracking
    the best genome across generations.

    Attributes:
        population (list[Genome]): Current population of genomes.
        bestFit (Genome | None): Best genome found after running `findSolution`.
    """
    def __init__(self, test: list[list[Union[float, int, str]]], variables: list[str], populationSize: int = 10, elitism: int = 0, mutationChance: float = 0.05, epochs: int = 10, error: float = 10e-5) -> 'GenProg':
        """
        Initializes a GenProg instance with population parameters and variables.

        Args:
            test (list[list[float | int | str]]): Table with first row as variable names 
                and subsequent rows containing variable values and expected results.
            variables (list[str]): Names of input variables.
            populationSize (int, optional): Size of the genome population. Must be > elitism. Defaults to 10.
            elitism (int, optional): Number of top genomes preserved across generations. Defaults to 0.
            mutationChance (float, optional): Mutation probability for each genome. Defaults to 0.05.
            epochs (int, optional): Number of generations to evolve. Defaults to 10.

        Raises:
            AttributeError: If `populationSize` <= `elitism` or `populationSize` < 10.
        """
        
        if populationSize <= elitism:
            raise AttributeError(f"Population size : {populationSize}, it must be larger the elitism : {elitism}")
        if populationSize < 10:
            raise AttributeError(f"Population size : {populationSize}, cant be less then 10")

        self.__test = test
        self.__populationSize = populationSize
        self.__variables = variables
        self.__elitism = elitism
        self.__mutationChance = mutationChance
        self.__epochs = epochs
        self.__error = error

        self.bestFit = None
        self.population = [Genome(self.__test, self.__variables, self.__mutationChance) for _ in range(self.__populationSize)]
        self.__newPopulation = [Genome(self.__test, self.__variables, self.__mutationChance) for _ in range(self.__populationSize)]

    def findSolution(self) -> list:
        """
        Evolves the population to find a genome approximating the target function.

        Applies selection, crossover, and mutation over multiple generations. Tracks
        the best genome's fitness at each generation.

        Returns:
            list[float]: List of best fitness values per generation.
        """
        startintTime = time.perf_counter()
        graph = []
        
        for i in range(self.__epochs):
            if i % 100 == 0:
                print(f"{i} : {time.perf_counter() - startintTime} s")

            for i in range(self.__populationSize):
                _ = self.population[i].getFitness()

            self.population.sort()

            graph.append(self.population[0].geneError)

            best_error = self.population[0].geneError
            if best_error is not None and np.isfinite(best_error) and best_error < self.__error:
                break
            
            self.__newPopulation[:self.__elitism] = self.population[:self.__elitism]
            
            for j in range(self.__elitism, self.__populationSize - 1, 2):
                idx1, idx2 = self.tournament()

                self.__newPopulation[j], self.__newPopulation[j+1] = self.crossover(self.population[idx1], self.population[idx2])

                self.__newPopulation[j].mutate()
                self.__newPopulation[j+1].mutate()
            
            self.population = self.__newPopulation
            
        best = min(self.population)
        self.bestFit = cp.deepcopy(best)
        print(f"elapse time : {time.perf_counter() - startintTime : .6f} s - solution: {best} - value: {best.gene.value} - fitness: {best.geneError}")
        return graph
    
    def crossover(self, genome1: 'Genome', genome2: 'Genome') -> list:
        """
        Performs subtree crossover between two parent genomes to create two offspring.

        Args:
            genome1 (Genome): First parent genome.
            genome2 (Genome): Second parent genome.

        Returns:
            list[Genome]: Two child genomes resulting from the crossover.
        """
        node1 = random.randint(1, genome1.gene.size)
        node2 = random.randint(1, genome2.gene.size)

        child1 = cp.deepcopy(genome1)
        child2 = cp.deepcopy(genome2)

        tree1 = child1.gene.subTree(node1)#uzmi stablo na poziciji node1
        tree2 = child2.gene.subTree(node2)
        _ = child1.gene.subTree(node1, cp.deepcopy(tree2))#zameni podstablo na poziciji node1 sa podstablom tree2
        _ = child2.gene.subTree(node2, cp.deepcopy(tree1))#uzmi stablo na poziciji node2
        
        return [child1, child2]
        
    def tournament(self) -> list:
        """
        Selects two genomes from the population using tournament selection.

        Chooses 10 random genomes and returns the indices of the two with the lowest fitness.

        Returns:
            list[int]: Indices of the two selected genomes.
        """
        indexes = random.sample(range(len(self.population)), 10)

        maxFit = float('inf')
        idx1 = indexes[0]
        idx2 = indexes[1]
        for i in indexes:
            if maxFit > self.population[i].geneError:
                maxFit = self.population[i].geneError
                idx2 = idx1
                idx1 = i
        
        return [idx1, idx2]
