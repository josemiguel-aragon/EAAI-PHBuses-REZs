from typing import List

from jmetal.core.operator import Mutation
from jmetal.core.operator import Crossover
from HybridBusSolution import HybridBusSolution

from jmetal.util.ckecking import Check
from jmetal.util.comparator import DominanceComparator

import random 
import copy

class HybridBusFlipMutation(Mutation[HybridBusSolution]):

    def __init__(self, probability: float):
        super(HybridBusFlipMutation, self).__init__(probability=probability)
    
    def execute(self, solution: HybridBusSolution) -> HybridBusSolution:
        Check.that(type(solution) is HybridBusSolution, "Solution type invalid")

        for i in range(solution.number_of_variables):
            for j in range(len(solution.variables[i])):
                rand = random.random()
                if rand <= self.probability:
                    section_percent = random.randint(0, 100)
                    solution.variables[i][j] = section_percent

        return solution

    def get_name(self):
        return "Hybrid Bus Flip mutation"

class HybridBusTPXCrossover(Crossover[HybridBusSolution, HybridBusSolution]):

    def __init__(self, probability: float):
        super(HybridBusTPXCrossover, self).__init__(probability=probability)

    def execute(self, parents: List[HybridBusSolution]) -> List[HybridBusSolution]:
        Check.that(type(parents[0]) is HybridBusSolution, "Solution type invalid")
        Check.that(type(parents[1]) is HybridBusSolution, "Solution type invalid")
        Check.that(len(parents) == 2, 'The number of parents is not two: {}'.format(len(parents)))

        offspring = [copy.deepcopy(parents[0]), copy.deepcopy(parents[1])]
        rand = random.random()

        if rand <= self.probability:
            # 1. Get the total number of variables
            total_number_of_variables = len(parents[0].variables[0])

            # 2. Calculate the points to make the crossover
            variable_to_cut1 = random.randrange(0, total_number_of_variables)
            variable_to_cut2 = random.randrange(0, total_number_of_variables)

            points = [variable_to_cut1, variable_to_cut2]
            points.sort()

            # 3. Apply the crossover to the variable
            for i in range(points[0] + 1, points[1] + 1):
                swap = offspring[0].variables[0][i]
                offspring[0].variables[0][i] = offspring[1].variables[0][i]
                offspring[1].variables[0][i] = swap

            # Get best parent solution
            dominance_comparator = DominanceComparator()
            result = dominance_comparator.compare(parents[0], parents[1])
            # First parent is the best one
            if result == -1:
                if len(range(points[0] + 1, points[1] + 1)) > (len(range(0,points[0] + 1)) +\
                    len(range(points[1] + 1, total_number_of_variables))):
                    swap = offspring[0]
                    offspring[0] =  offspring[1]
                    offspring[1] = swap
            # Second parent is the best one
            elif result == 1:
                if not (len(range(points[0] + 1, points[1] + 1)) > (len(range(0,points[0] + 1)) + \
                 len(range(points[1] + 1, total_number_of_variables)))):
                    swap = offspring[0]
                    offspring[0] =  offspring[1]
                    offspring[1] = swap            
                
        return offspring

    def get_number_of_parents(self) -> int:
        return 2
    
    def get_number_of_children(self) -> int:
        return 2

    def get_name(self) -> str:
        return 'Integer Two point crossover'