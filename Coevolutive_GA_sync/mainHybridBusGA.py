
from HybridBusOperators import HybridBusFlipMutation, HybridBusTPXCrossover
from HybridBusProblem import HybridBusProblem
from HybridBusSolution import HybridBusSolution

from jmetal.algorithm.multiobjective.mocell import MOCell
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.operator.selection import BinaryTournamentSelection
from jmetal.util.archive import CrowdingDistanceArchive
from jmetal.util.evaluator import MultiprocessEvaluator
from jmetal.util.solution import get_non_dominated_solutions
from jmetal.lab.visualization import Plot
from jmetal.util.neighborhood import C9
from jmetal.util.observer import ProgressBarObserver, WriteFrontToFileObserver

from CustomObservers import NonDomWriteFrontToFileObserver, NonDomPlotFrontToFileObserver
from ReadRoute import read_route
from Bus import Bus

import pandas as pd
import logging

config_route_path = '../output/processed_bus_route_M6_random_15.0%ze.csv'

#Settings
df = pd.read_csv(config_route_path, sep=',', index_col=0, header=None)
route = read_route(config_route_path)

config_max_evaluations = 1000000 # Maximum number of evaluations for the GA
config_population_size = 100 # Total population size of the GA
config_offspring_population_size = 100 # Total offspring population size of the GA
config_probability_mutation = 1. / len(route.sections) # Mutation probability for the GA
config_probability_crossover = 1 # Crossover probability for the GA
config_crossover_operator = HybridBusTPXCrossover(config_probability_crossover)
config_mutation_operator = HybridBusFlipMutation(config_probability_mutation)
config_selection_operator = BinaryTournamentSelection()
config_neighborhood = C9(10, 10)
config_archive = CrowdingDistanceArchive(config_population_size)

LOGGER = logging.getLogger("jmetal")
LOGGER.disabled = True

if __name__ == '__main__':

    main_bus = Bus(1, route)

    # Problem set
    problem = HybridBusProblem (route = route, bus = main_bus)

    algorithm = MOCell(problem = problem,
        population_size = config_population_size,
        neighborhood = config_neighborhood,
        archive = config_archive,
        population_evaluator = MultiprocessEvaluator(processes = 1),
        mutation = config_mutation_operator,
        crossover= config_crossover_operator,
        termination_criterion = StoppingByEvaluations(max_evaluations=config_max_evaluations))

    # Setting algorithm observers
    #algorithm.observable.register(WriteFrontToFileObserver("./generation_front_files"))
    #algorithm.observable.register(NonDomWriteFrontToFileObserver("./generation_nondom_front_files"))
    #algorithm.observable.register(NonDomPlotFrontToFileObserver("./generation_nondom_front_plots"))
    algorithm.observable.register(ProgressBarObserver(max = config_max_evaluations))


    # Run genetic algorithm
    algorithm.run()

    with open("results.data","w") as file:
        #Outputs
        file.write('\nSettings:')
        file.write(f'\n\tAlgorithm: {algorithm.get_name()}')
        file.write(f'\n\tProblem: {problem.get_name()}')
        file.write(f'\n\tComputing time: {algorithm.total_computing_time} seconds')
        file.write(f'\n\tMax evaluations: {config_max_evaluations}')
        file.write(f'\n\tPopulation size: {config_population_size}')
        file.write(f'\n\tOffspring population size: {config_offspring_population_size}')
        file.write(f'\n\tProbability mutation: {config_probability_mutation}')
        file.write(f'\n\tProbability crossover: {config_probability_crossover}')
        file.write(f'\n\tSolution length: {len(route.sections)}')
        file.write('\nResults:')
        for solution in algorithm.get_result():
            file.write(f'\n\tBest solution: {solution.variables}')
            file.write(f'\n\tFitness: [{solution.objectives[0]}, {solution.objectives[1]}]')
            file.write('\n\tPartners:')
            for partner in solution.partners:
                file.write(f'\n\t\t{partner.variables}')

    with open("nondom_results.data","w") as file:
        #Outputs
        file.write('\nSettings:')
        file.write(f'\n\tAlgorithm: {algorithm.get_name()}')
        file.write(f'\n\tProblem: {problem.get_name()}')
        file.write(f'\n\tComputing time: {algorithm.total_computing_time} seconds')
        file.write(f'\n\tMax evaluations: {config_max_evaluations}')
        file.write(f'\n\tPopulation size: {config_population_size}')
        file.write(f'\n\tOffspring population size: {config_offspring_population_size}')
        file.write(f'\n\tProbability mutation: {config_probability_mutation}')
        file.write(f'\n\tProbability crossover: {config_probability_crossover}')
        file.write(f'\n\tSolution length: {len(route.sections)}')
        file.write('\nResults:')
        for solution in get_non_dominated_solutions(algorithm.get_result()):
            file.write(f'\n\tBest solution: {solution.variables}')
            file.write(f'\n\tFitness: [{solution.objectives[0]}, {solution.objectives[1]}]')
            file.write('\n\tPartners:')
            for partner in solution.partners:
                file.write(f'\n\t\t{partner.variables}')
            

    print('\nSettings:')
    print(f'\tAlgorithm: {algorithm.get_name()}')
    print(f'\tProblem: {problem.get_name()}')
    print(f'\tComputing time: {algorithm.total_computing_time} seconds')
    print(f'\tMax evaluations: {config_max_evaluations}')
    print(f'\tPopulation size: {config_population_size}')
    print(f'\tOffspring population size: {config_offspring_population_size}')
    print(f'\tProbability mutation: {config_probability_mutation}')
    print(f'\tProbability crossover: {config_probability_crossover}')
    print(f'\tSolution length: {len(route.sections)}')
    print('\nResults:')
    for solution in get_non_dominated_solutions(algorithm.get_result()):
        print(f'\tBest solution: {solution.variables}')
        print(f'\tFitness: [{solution.objectives[0]}, {solution.objectives[1]}]')
        print('\tPartners:')
        for partner in solution.partners:
            print(f'\t\t{partner.variables}')
