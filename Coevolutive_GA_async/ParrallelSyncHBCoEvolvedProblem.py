from HybridBusProblemCoEv import HybridBusProblem
from Route import Route
from Bus import Bus
from ReadRoute import read_route

from jmetal.algorithm.multiobjective.mocell import MOCell
from jmetal.util.solution import get_non_dominated_solutions
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.operator.selection import BinaryTournamentSelection
from jmetal.operator.crossover import TPXCrossover
from jmetal.operator.mutation import BitFlipMutation
from jmetal.util.archive import CrowdingDistanceArchive
from jmetal.util.evaluator import MultiprocessEvaluator
from jmetal.lab.visualization import Plot
from jmetal.util.neighborhood import C9
from jmetal.util.observer import ProgressBarObserver, WriteFrontToFileObserver
from jmetal.util.constraint_handling import feasibility_ratio
from CustomObservers import NonDomWriteFrontToFileObserver

from random import randint
from tqdm import tqdm
from time import time
import pandas as pd
import os
import sys
import logging

from multiprocessing import Process, Manager, Barrier
from multiprocessing.shared_memory import ShareableList

LOGGER = logging.getLogger("jmetal")
LOGGER.disabled = True

def run_problem(process_id, problem, config_population_size, config_neighborhood,
                config_archive, config_mutation_operator, config_crossover_operator,
                config_max_evaluations, barrier, representatives,
                config_offspring_population_size):
    # Create algorithm
    algorithm = MOCell(problem = problem,
                population_size = config_population_size,
                neighborhood = config_neighborhood,
                archive = config_archive,
                mutation = config_mutation_operator(1.0 / len(problem.route.sections)),
                crossover= config_crossover_operator,
                termination_criterion = StoppingByEvaluations(max_evaluations=config_max_evaluations))
    algorithm.observable.register(WriteFrontToFileObserver(f"./generation_front_files_process_{process_id}"))
    #algorithm.observable.register(NonDomWriteFrontToFileObserver(f"./nondom_generation_front_files_process_{process_id}"))

    # Initialize population
    algorithm.solutions = algorithm.create_initial_solutions()

    #################
    #SYNCHRONIZATION
    #################

    repre = algorithm.solutions[randint(0, config_population_size - 1)]
    for index,_ in enumerate(representatives[process_id]):
        representatives[process_id][index] = repre.variables[index]

    barrier.wait()
    algorithm.init_step()
    barrier.wait()

    evaluations = config_population_size
    epochs = 1
    print(f"Process {process_id}: EPOCH {epochs}")

    while evaluations < config_max_evaluations:
        # Iteration i (t=1)
        algorithm.run_step()

        evaluations += config_offspring_population_size

        if algorithm.current_individual == 0:
            epochs += 1
            #################
            #SYNCHRONIZATION
            #################
            barrier.wait()

            actual_front = algorithm.get_result()
            repre = actual_front[randint(0, len(actual_front) - 1)]
            for index,_ in enumerate(representatives[process_id]):
                representatives[process_id][index] = repre.variables[index]

            repres = actual_front

            print(f"Process {process_id}: EPOCH {epochs}")




class ParallelSyncHBCoEvolvedProblem:
    def __init__(self, routes, buses):
        self.barrier = Barrier(len(routes))

        self.routes = routes
        self.buses = buses
        self.representatives = []

        self.config_max_evaluations = 100000 # Maximum number of evaluations for the GA
        self.config_population_size = 100 # Total population size of the GA
        self.config_offspring_population_size = 1 # Total offspring population size of the GA
        self.config_probability_crossover = 1 # Crossover probability for the GA
        self.config_crossover_operator = TPXCrossover(self.config_probability_crossover)
        self.config_mutation_operator = BitFlipMutation
        self.config_selection_operator = BinaryTournamentSelection()
        self.config_neighborhood = C9(10, 10)
        self.config_archive = CrowdingDistanceArchive(self.config_population_size)
        self.problems = []
        for index, route in enumerate(self.routes):
            self.problems.append(HybridBusProblem (route = route, bus = self.buses[index], process_id = index, representatives = self.representatives))
            self.representatives.append(ShareableList([False] * self.problems[-1].number_of_variables))
            self.problems[-1].representatives = self.representatives
            self.problems[-1].single_compute_green_zones_maximums()
        for problem in self.problems:
            problem.problems = self.problems
        self.problems[-1].total_compute_green_zones_maximums()
        self.problems[-1].calculate_total_greenK()
        print("AAACEITE")


    def run(self):

        begin = time()
        ######################################
        # Launch processes
        self.processes = []

        for index, route in enumerate(self.routes):
            p = Process(target=run_problem, args=(index, self.problems[index],  self.config_population_size, self.config_neighborhood,
                                                  self.config_archive, self.config_mutation_operator, self.config_crossover_operator,
                                                  self.config_max_evaluations, self.barrier, self.representatives,
                                                  self.config_offspring_population_size))
            p.start()
            self.processes.append(p)


        for p in self.processes:
            p.join()

        # Pareto Front reconstruction
        #pareto_front = []
        #for index, results in enumerate(self.representatives):
        #     pareto_front += get_non_dominated_solutions(results)
        ############################################
        end = time()

        with open("./elapsed_time.txt","w+") as file:
            file.write(f"Tiempo total: {(end-begin)} segundos")
        #return get_non_dominated_solutions(pareto_front)

if __name__ == '__main__':
    config_route_paths = os.listdir('../output/')

    #Settings
    routes = []
    buses = []
    for path in config_route_paths:
        if ".csv" in path:
            df = pd.read_csv('../output/'+path, sep=',', index_col=0, header=None)
            routes.append(read_route('../output/'+path))
            buses.append(Bus(1, routes[-1]))

    problem = ParallelSyncHBCoEvolvedProblem(routes, buses)

    problem.run()
    '''
    with open("results.data","w") as file:
        #Outputs
        file.write('\nResults:')
        for solution in front:
            file.write(f'\n\tBest solution: {solution.variables}')
            file.write(f'\n\tFitness: [{solution.objectives[0]}, {solution.objectives[1]}]')
            file.write('\n\tPartners:')
            for partner in solution.partners:
                if partner == None:
                    file.write(f'\n\t\t{partner}')
                else:
                    file.write(f'\n\t\t{partner.variables}')

    with open("nondom_results.data","w") as file:
        #Outputs
        file.write('\nResults:')
        for solution in get_non_dominated_solutions(front):
            file.write(f'\n\tBest solution: {solution.variables}')
            file.write(f'\n\tFitness: [{solution.objectives[0]}, {solution.objectives[1]}]')
            file.write('\n\tPartners:')
            for partner in solution.partners:
                if partner == None:
                    file.write(f'\n\t\t{partner}')
                else:
                    file.write(f'\n\t\t{partner.variables}')

    print('\nResults:')
    for solution in get_non_dominated_solutions(front):
        print(f'\tBest solution: {solution.variables}')
        print(f'\tFitness: [{solution.objectives[0]}, {solution.objectives[1]}]')
        print('\tPartners:')
        for partner in solution.partners:
            if partner == None:
                print(f'\t\t{partner}')
            else:
                print(f'\t\t{partner.variables}')
        '''


