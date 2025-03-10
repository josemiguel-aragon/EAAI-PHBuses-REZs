from typing import TypeVar, List

from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.config import store
from jmetal.core.operator import Mutation, Crossover, Selection
from jmetal.core.problem import Problem
from jmetal.operator import BinaryTournamentSelection
from jmetal.util.comparator import MultiComparator
from jmetal.util.density_estimator import CrowdingDistance
from jmetal.util.evaluator import Evaluator
from jmetal.util.generator import Generator
from jmetal.util.neighborhood import Neighborhood
from jmetal.util.ranking import FastNonDominatedRanking
from jmetal.util.termination_criterion import TerminationCriterion

S = TypeVar('S')
R = TypeVar('R')

"""
.. module:: Cellular Genetic Algorithm
   :platform: Unix, Windows
   :synopsis: Cellular Genetic Algorithm (cGA) implementation
.. moduleauthor:: Jose M. Aragon-Jurado
"""


class CellularGeneticAlgorithm(GeneticAlgorithm[S, R]):

    def __init__(self,
                 problem: Problem,
                 population_size: int,
                 neighborhood: Neighborhood,
                 mutation: Mutation,
                 crossover: Crossover,
                 selection: Selection = BinaryTournamentSelection(
                     MultiComparator([FastNonDominatedRanking.get_comparator(),
                                      CrowdingDistance.get_comparator()])),
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator
                ):
        """
        cGA implementation as described in:

        :param problem: The problem to solve.
        :param population_size: Size of the population.
        :param mutation: Mutation operator (see :py:mod:`jmetal.operator.mutation`).
        :param crossover: Crossover operator (see :py:mod:`jmetal.operator.crossover`).
        :param selection: Selection operator (see :py:mod:`jmetal.operator.selection`).
        """
        super(CellularGeneticAlgorithm, self).__init__(
            problem=problem,
            population_size=population_size,
            offspring_population_size=1,
            mutation=mutation,
            crossover=crossover,
            selection=selection,
            termination_criterion=termination_criterion,
            population_evaluator=population_evaluator,
            population_generator=population_generator
        )
        self.neighborhood = neighborhood
        self.current_individual = 0
        self.current_neighbors = []

    def update_progress(self) -> None:
        self.evaluations += 1
        
        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)

        self.current_individual = (self.current_individual + 1) % self.population_size

        # Save current population and current best solution
        if self.current_individual == 0:
            with open('./ga_progress.data', 'a+') as result:
                result.write('## EPOCH {} ##\n'.format(self.problem.epoch))
                result.write('Population: \n')
                for sol in self.solutions:
                    result.write('\tSolution: {}\n'.format(sol.variables))
                    result.write('\tFitness: {}\n'.format(sol.objectives[0]))
                result.write('BEST SOLUTION:\n')
                result.write('\tSolution: {}\n'.format(self.get_result().variables))
                result.write('\tFitness: {}\n'.format(self.get_result().objectives[0]))

    def selection(self, population: List[S]):
        parents = []

        self.current_neighbors = self.neighborhood.get_neighbors(self.current_individual, population)
        self.current_neighbors.append(self.solutions[self.current_individual])
        
        
        p1 = self.selection_operator.execute(self.current_neighbors)
        self.current_neighbors.remove(p1)
        p2 = self.selection_operator.execute(self.current_neighbors)
        
        parents = parents + [p1,p2]

        return parents

    def reproduction(self, mating_population: List[S]) -> List[S]:
        number_of_parents_to_combine = self.crossover_operator.get_number_of_parents()
        if len(mating_population) % number_of_parents_to_combine != 0:
            raise Exception('Wrong number of parents')

        offspring_population = self.crossover_operator.execute(mating_population)
        self.mutation_operator.execute(offspring_population[0])

        return [offspring_population[0]]

    def replacement(self, population: List[S], offspring_population: List[S]) -> List[List[S]]:
        if population[self.current_individual].objectives[0] > offspring_population[0].objectives[0]: # Check if new solution is better
            population[self.current_individual] = offspring_population[0]
            
        return population

    def get_result(self) -> R:
        return min(self.solutions,key=lambda s: s.objectives[0])

    def get_name(self) -> str:
        return 'cGA'
