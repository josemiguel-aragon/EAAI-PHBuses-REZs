from jmetal.core.solution import Solution

class HybridBusSolution(Solution):

    """Class representing Hybrid Bus Problem solutions"""

    def __init__(self, number_of_variables: int = 1, number_of_objectives: int = 2, number_of_constraints: int = 0):
        super(HybridBusSolution, self).__init__(number_of_variables, number_of_objectives, number_of_constraints)

        self.variables = [[] for _ in range(self.number_of_variables) ]
    
    def __copy__(self):
        new_solution = HybridBusSolution(self.number_of_variables, self.number_of_objectives)
        new_solution.objectives = self.objectives[:]
        new_solution.variables = self.variables[:]

        new_solution.attributes = self.attributes.copy()

        return new_solution

    def n_battery_sections(self) -> int:
        count = 0
        for var in self.variables:
            for section in var:
                if section != 0:
                    count += 1
        return count
    
    def remaining_charge(self) -> int:
        charge = 0
        for var in self.variables:
            for section_charge in var:
                charge += section_charge
        return charge