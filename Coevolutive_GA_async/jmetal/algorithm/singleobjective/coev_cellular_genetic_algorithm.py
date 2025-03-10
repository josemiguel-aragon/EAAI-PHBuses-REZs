import threading, time, random
import logging


from typing import TypeVar, List, Generic

from jmetal.core.algorithm import Algorithm

from jmetal.config import store
from jmetal.util.evaluator import CoevolvedSequentialEvaluator
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

LOGGER = logging.getLogger('jmetal')

S = TypeVar('S')
R = TypeVar('R')

"""
.. module:: Coevolved Cellular Genetic Algorithm
   :platform: Unix, Windows
   :synopsis: Coevolved Cellular Genetic Algorithm (Co-cGA) implementation
.. moduleauthor:: Jose M. Aragon-Jurado
"""


class CoevolvedCellularGeneticAlgorithm(Generic[S, R], threading.Thread):

    def __init__(self,
                problem1: Problem,
                problem2: Problem,
                population_size: int,
                neighborhood: Neighborhood,
                mutation: Mutation,
                crossover: Crossover,
                selection: Selection = BinaryTournamentSelection(
                MultiComparator([FastNonDominatedRanking.get_comparator(),
                                CrowdingDistance.get_comparator()])),
                termination_criterion: TerminationCriterion = store.default_termination_criteria,
                population_generator: Generator = store.default_generator,
                population_evaluator: Evaluator = CoevolvedSequentialEvaluator()
                ):
        """
        coop-cGA implementation as described in:

        :param problem1: The first problem to solve.
        :param problem2: The second problem to solve.

        :param population_size: Size of both populations.

        :param neighborhood: Neighborhood criterion (see :py:mod:`jmetal.util.neighborhood`).
        :param mutation: Mutation operator (see :py:mod:`jmetal.operator.mutation`).
        :param crossover: Crossover operator (see :py:mod:`jmetal.operator.crossover`).
        :param selection: Selection operator (see :py:mod:`jmetal.operator.selection`).
        """
        
        threading.Thread.__init__(self)

        self.solutions1: List[S] = []
        self.solutions2: List[S] = []

        self.evaluations = 0
        self.start_computing_time = 0
        self.total_computing_time = 0

        self.observable = store.default_observable

        self.problem1 = problem1
        self.problem2 = problem2
        
        self.population_size = population_size
        
        self.mutation_operator = mutation
        self.crossover_operator = crossover
        self.selection_operator = selection
        

        self.population_generator = population_generator
        self.population_evaluator = population_evaluator

        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)

        self.mating_pool_size = \
            1 * \
            self.crossover_operator.get_number_of_parents() // self.crossover_operator.get_number_of_children()

        if self.mating_pool_size < self.crossover_operator.get_number_of_children():
            self.mating_pool_size = self.crossover_operator.get_number_of_children()
        
        self.neighborhood = neighborhood

        self.current_individual = 0
        self.current_neighbors = []

    def create_initial_solutions(self) -> List[S]:
        """ Creates the initial list of solutions of a metaheuristic. """
        return [[self.population_generator.new(self.problem1)
                for _ in range(self.population_size)], 
                
                [self.population_generator.new(self.problem2)
                for _ in range(self.population_size)]]
    
    def run(self):
        """ Execute the algorithm. """
        self.start_computing_time = time.time()

        self.solutions1, self.solutions2 = self.create_initial_solutions()

        self.best_problem1 = self.solutions1[random.randrange(self.population_size)]
        self.best_problem2 = self.solutions2[random.randrange(self.population_size)]
        
        self.solutions1 = self.evaluate(self.solutions1, self.problem1, self.best_problem2)
        self.solutions2 = self.evaluate(self.solutions2, self.problem2, self.best_problem1)

        LOGGER.debug('Initializing progress')
        self.init_progress()

        LOGGER.debug('Running main loop until termination criteria is met')
        while not self.stopping_condition_is_met():
            self.current_individual = 0

            self.best_problem1 = min(self.solutions1,key=lambda s: s.objectives[0])
            self.best_problem2 = min(self.solutions2,key=lambda s: s.objectives[0])

            while self.current_individual < self.population_size:
                self.step(self.solutions1, self.problem1, self.best_problem2)
                self.current_individual += 1

            self.current_individual = 0
            while self.current_individual < self.population_size:
                self.step(self.solutions2, self.problem2, self.best_problem1)
                self.current_individual += 1

            self.update_progress()

        self.total_computing_time = time.time() - self.start_computing_time

    
    def evaluate(self, population: List[S], problem: Problem, problem2_individual: S):
        return self.population_evaluator.evaluate(population, problem, problem2_individual)

    def init_progress(self) -> None:
        self.evaluations = self.population_size

        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)

    def stopping_condition_is_met(self) -> bool:
        return self.termination_criterion.is_met

    def step(self, solutions: List[S], problem: Problem, problem2_individual: S):
        mating_population = self.selection(solutions)
        offspring_population = self.reproduction(mating_population)
        offspring_population = self.evaluate(offspring_population, problem, problem2_individual)

        return self.replacement(solutions, offspring_population)

    
    def selection(self, population: List[S]):
        parents = []

        self.current_neighbors = self.neighborhood.get_neighbors(self.current_individual, population)
        self.current_neighbors.append(population[self.current_individual])
        
        
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
    

    def update_progress(self) -> None:
        self.evaluations += self.population_size
        self.problem1.epoch += 1
        
        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)

        with open('./ga_progress.data', 'a+') as result:
            result.write('## EPOCH {} ##\n'.format(self.problem1.epoch))
            result.write('Problem 1: \n')
            result.write('\tPopulation: \n')
            for sol in self.solutions1:
                result.write('\t\tSolution: {}\n'.format(sol.variables))
                result.write('\t\tFitness: {}\n'.format(sol.objectives[0]))
            result.write('Problem 2: \n')
            result.write('\tPopulation: \n')
            for sol in self.solutions2:
                result.write('\t\tSolution: {}\n'.format(sol.variables))
                result.write('\t\tFitness: {}\n'.format(sol.objectives[0]))
            result.write('BEST SOLUTION:\n')
            result.write('\tSolution: {}\n {}\n'.format(self.get_result()[0].variables, self.get_result()[1].variables))
            result.write('\tFitness: {}\n'.format(self.get_result()[2]))
            
            
    def get_observable_data(self) -> dict:
        return {'PROBLEM': self.problem1,
                'EVALUATIONS': self.evaluations,
                'SOLUTIONS': self.get_result(),
                'COMPUTING_TIME': time.time() - self.start_computing_time}

    def get_result(self) -> R:
        best_sol1 = min(self.solutions1,key=lambda s: s.objectives[0])
        best_sol2 = min(self.solutions1,key=lambda s: s.objectives[0])
         
        if best_sol1.objectives[0] < best_sol2.objectives[0]:
            return [best_sol1, self.best_problem2, best_sol1.objectives[0]]
        else:
            return [self.best_problem1, best_sol2, best_sol2.objectives[0]]

    def get_name(self) -> str:
        return 'Coevolved CGA'
