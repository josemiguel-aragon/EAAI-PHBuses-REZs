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



class HBCoEvolvedProblem:
    def __init__(self, routes: list[Route], buses: list[Bus]):
        self.config_max_evaluations = 1000000 # Maximum number of evaluations for the GA
        self.config_population_size = 100 # Total population size of the GA
        self.config_offspring_population_size = 1 # Total offspring population size of the GA
        self.config_probability_crossover = 1 # Crossover probability for the GA
        self.config_crossover_operator = HybridBusTPXCrossover(self.config_probability_crossover)
        self.config_mutation_operator = HybridBusFlipMutation
        self.config_selection_operator = BinaryTournamentSelection()
        self.config_neighborhood = C9(10, 10)
        self.config_archive = CrowdingDistanceArchive(self.config_population_size)
        self.representatives = []
        
        self.routes = routes
        self.buses = buses

            
            
    def run(self):
        begin = time()
        ######################################
        self.subproblems = [HybridBusProblem (route = route, bus = self.buses[index], process_id = index, representatives=self.representatives) for index,route in enumerate(self.routes)]
         
        self.subalgorithms = []
        
        for problem in self.subproblems:
            algorithm = MOCell(problem = problem,
                            population_size = self.config_population_size,
                            neighborhood = self.config_neighborhood,
                            archive = self.config_archive,
                            mutation = self.config_mutation_operator(1.0 / len(problem.route.sections)),
                            crossover= self.config_crossover_operator,
                            termination_criterion = StoppingByEvaluations(max_evaluations=self.config_max_evaluations))
            algorithm.solutions = algorithm.create_initial_solutions()
            self.representatives.append(algorithm.solutions[randint(0, self.config_population_size)])
            self.subalgorithms.append(algorithm)
            
        # First iteration (t=0) Coevolved process has not started yet
        for index, algorithm in enumerate(self.subalgorithms):
            algorithm.init_step()
        
        self.epoch = 1
        self.evaluations = self.config_population_size
        
        #with tqdm(total=self.config_max_evaluations) as pbar:
        while self.evaluations < self.config_max_evaluations:
            
            # Iteration i (t=1)
            for index, algorithm in enumerate(self.subalgorithms):
                algorithm.run_step()
                self.representatives[index] = algorithm.get_result()
                
            self.epoch += 1
            self.evaluations += self.config_offspring_population_size
                #pbar.update(self.config_offspring_population_size)

        
        # Pareto Front reconstruction
        #pareto_front = []
        #for index, algorithm in enumerate(self.subalgorithms):
        #     pareto_front += get_non_dominated_solutions(algorithm.get_result())
             
        end = time()
        ############################################
        print(f"Tiempo total: {(end-begin)} segundos")
        
        #return get_non_dominated_solutions(pareto_front)


if __name__ == '__main__'
    config_route_paths = [
        '../output/processed_7_parades_linia_Barcelona_conAlturas_green_zones.csv',
        '../output/processed_22_parades_linia_Barcelona_conAlturas_green_zones.csv',
        '../output/processed_H10_parades_linia_Barcelona_conAlturas_green_zones.csv',
        '../output/processed_H12_parades_linia_Barcelona_conAlturas_green_zones.csv',
        '../output/processed_V15_parades_linia_Barcelona_conAlturas_green_zones.csv',
        '../output/processed_V17_parades_linia_Barcelona_conAlturas_green_zones.csv'
    ]

    #Settings
    routes = []
    buses = []
    for path in config_route_paths:
        
        df = pd.read_csv(path, sep=',', index_col=0, header=None)
        routes.append(read_route(path))
        buses.append(Bus(1, routes[-1]))
        
    problem = HBCoEvolvedProblem(routes, buses)
    
    problem.run()
        