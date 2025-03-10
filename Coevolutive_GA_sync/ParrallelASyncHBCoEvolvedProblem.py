from HybridBusProblemCoEv import HybridBusProblem
from HybridBusOperators import HybridBusFlipMutation, HybridBusTPXCrossover
from Route import Route
from Bus import Bus
from ReadRoute import read_route

from jmetal.algorithm.multiobjective.mocell import MOCell
from jmetal.util.solution import get_non_dominated_solutions
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.operator.selection import BinaryTournamentSelection
from jmetal.util.archive import CrowdingDistanceArchive
from jmetal.lab.visualization import Plot
from jmetal.util.neighborhood import C9
from jmetal.util.observer import ProgressBarObserver, WriteFrontToFileObserver

from random import randint
from tqdm import tqdm
from time import time
import pandas as pd

from multiprocessing import Process, Manager, Barrier

def run_problem(process_id, route, bus, config_population_size, config_neighborhood,
                config_archive, config_mutation_operator, config_crossover_operator,
                config_max_evaluations, barrier, representatives,  
                config_offspring_population_size):
    # Create algorithm
    problem = HybridBusProblem (route = route, bus = bus, process_id = process_id, representatives = representatives)
    
    algorithm = MOCell(problem = problem,
                population_size = config_population_size,
                neighborhood = config_neighborhood,
                archive = config_archive,
                mutation = config_mutation_operator(1.0 / len(problem.route.sections)),
                crossover= config_crossover_operator,
                termination_criterion = StoppingByEvaluations(max_evaluations=config_max_evaluations))
    
    # Initialize population
    algorithm.solutions = algorithm.create_initial_solutions()
    
    #################
    #SYNCHRONIZATION
    #################
    representatives[process_id]=algorithm.solutions[randint(0, config_population_size - 1)]
    barrier.wait()
    algorithm.init_step()
    barrier.wait()
    
    
    evaluations = config_population_size
    epochs = 1
    
    while evaluations < config_max_evaluations:
        # Iteration i (t=1)
        algorithm.run_step()
        
        #################
        #SYNCHRONIZATION
        #################
        representatives[process_id] = algorithm.get_result()
        
        evaluations += config_offspring_population_size
        
        if evaluations % config_population_size == 0:
            epochs += 1



class ParallelSyncHBCoEvolvedProblem:
    def __init__(self, routes: list[Route], buses: list[Bus]):
        self.manager = Manager()
        self.representatives = self.manager.list()
        self.barrier = Barrier(len(routes))
            
        self.routes = routes
        self.buses = buses
        
        for route in self.routes:
            self.representatives.append([])
        
        self.config_max_evaluations = 1000000 # Maximum number of evaluations for the GA
        self.config_population_size = 100 # Total population size of the GA
        self.config_offspring_population_size = 1 # Total offspring population size of the GA
        self.config_probability_crossover = 1 # Crossover probability for the GA
        self.config_crossover_operator = HybridBusTPXCrossover(self.config_probability_crossover)
        self.config_mutation_operator = HybridBusFlipMutation
        self.config_selection_operator = BinaryTournamentSelection()
        self.config_neighborhood = C9(10, 10)
        self.config_archive = CrowdingDistanceArchive(self.config_population_size)

                
    def run(self):
        begin = time()
        ######################################
        # Launch processes
        self.processes = []
        
        for index, route in enumerate(self.routes):
            p = Process(target=run_problem, args=(index, route, self.buses[index],  self.config_population_size, self.config_neighborhood,
                                                  self.config_archive, self.config_mutation_operator, self.config_crossover_operator,
                                                  self.config_max_evaluations, self.barrier, self.representatives,  
                                                  self.config_offspring_population_size))
            p.start()
            self.processes.append(p)
        
        for p in self.processes:
            p.join()
        
        # Pareto Front reconstruction
        pareto_front = []
        for index, results in enumerate(self.representatives):
             pareto_front += get_non_dominated_solutions(results)
        ############################################
        end = time()
        
        print(f"Tiempo total: {(end-begin)} segundos")
        return get_non_dominated_solutions(pareto_front)

if __name__ == '__main__':
    config_route_path = '../output/processed_bus_route_M6_random_0.0%ze.csv'

    #Settings
    df = pd.read_csv(config_route_path, sep=',', index_col=0, header=None)
    route = read_route(config_route_path)
    
    problem = ParallelSyncHBCoEvolvedProblem([route, route], [Bus(1, route), Bus(1, route)])
    #HBCoEvolvedProblem([route], [Bus(1, route)])
    
    print(problem.run())
        