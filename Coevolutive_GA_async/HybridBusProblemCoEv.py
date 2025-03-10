from jmetal.core.problem import Problem
from jmetal.core.solution import BinarySolution
from Route import Route
from Bus import Bus

import random
import math
import time
import numpy as np
import copy



def vehicle_specific_power(v: float, slope: float, acc: float, m: float = 19500, A: float = 7):
    g = 9.80665 # Gravity
    Cr=0.013 # Rolling resistance coefficient
    Cd=0.7 # Drag coefficient
    ro_air=1.225 # Air density
    alpha = math.atan(slope) #Elevation angle in radians
    aux_energy = 2 #Auxiliar energy consumption in kW - 3kW with AC off or 12kW with AC on

    #Vehicle efficiencies
    n_dc = 0.90
    n_m = 0.95
    n_t = 0.96
    n_b = 0.97
    n_g = 0.90

    Frff = g * Cr * m * math.cos(alpha) # Rolling friction force
    Fadf = ro_air * A * Cd * math.pow(v, 2) / 2 # Aerodynamic drag force
    Fhcf = g * m * math.sin(alpha) # Hill climbing force
    Farf = m * acc # Acceleration resistance force
    Fttf = Frff + Fadf + Fhcf + Farf # Total force in Newtons

    power = (Fttf * v) / 1000 # Total energy in kW

    #Drivetrain model (efficiency)
    rbf = 1-math.exp(-v*0.36) #Regenerative braking factor
    if power<0:
        total_n = n_dc*n_g*n_t*n_b #Total drivetrain efficiency
        total_power = aux_energy/n_b + rbf*power*total_n
    else:
        total_n = n_dc*n_m*n_t*n_b; #Total drivetrain efficiency
        total_power = aux_energy/n_b + power/total_n


    #print("Frff: {}, Fadf: {}, Fhcf: {}".format(Frff, Fadf, Fhcf))
    return  total_power

def decrease_battery_charge(remaining_charge, section_charge, bus_charge):
    if (remaining_charge - section_charge) > bus_charge:
        return bus_charge
    else:
        return remaining_charge - section_charge

def acceleration_section_power(vo: float, vf: float, acc: float, slope: float, section_distance: float,
                                section_duration: float, green_percent: float, m: float = 19500, A: float = 7):
    # Hasta los 25 m o 15km/h siempre en eléctrico
    constant_speed = 4.1666667

    acc_green_energies = 0
    acc_fuel_energies = 0


    instant_speed = 0
    acc_distance = (math.pow(vf, 2) - math.pow(vo, 2)) / (2 * acc)
    acc_duration = round((vf - vo) / acc)


    driven_distance = 0

    section_distance *= 1000
    green_distance = green_percent * section_distance

    if green_distance == section_distance:
        for _ in range(0, acc_duration):
            acc_green_energies += vehicle_specific_power(instant_speed, slope, acc, m, A) / 3600
            instant_speed += acc
        remaining_seconds = section_duration - acc_duration
        return acc_green_energies + vehicle_specific_power(vf, slope, 0, m, A) * remaining_seconds / 3600, [0], acc_green_energies
    elif green_distance > acc_distance:
        for _ in range(0, acc_duration):
            acc_green_energies += vehicle_specific_power(instant_speed, slope, acc, m, A) / 3600
            instant_speed += acc
        remaining_seconds = section_duration - acc_duration
        remaining_distance = section_distance - acc_distance

        remaining_green_percent = (green_distance - acc_distance)/ remaining_distance

        return acc_green_energies + vehicle_specific_power(vf, slope, acc, m, A) * remaining_seconds / 3600 * remaining_green_percent,\
            [vehicle_specific_power(vf, slope, acc, m, A) * remaining_seconds / 3600 * (1 - remaining_green_percent)],\
            acc_green_energies
    else:
        if vf >= constant_speed:
            for _ in range(0, acc_duration):
                if green_distance > driven_distance or instant_speed < constant_speed:
                    acc_green_energies += vehicle_specific_power(instant_speed, slope, acc, m, A) / 3600
                    driven_distance = (math.pow(instant_speed, 2) - math.pow(0, 2)) / (2 * acc)
                    instant_speed += acc
                else:
                    acc_fuel_energies += vehicle_specific_power(instant_speed, slope, acc, m, A) / 3600
                    instant_speed += acc
            remaining_seconds = section_duration - acc_duration
        else:
            for _ in range(0, acc_duration):
                if green_distance > driven_distance or driven_distance < 25:
                    acc_green_energies += vehicle_specific_power(instant_speed, slope, acc, m, A) / 3600
                    driven_distance = (math.pow(instant_speed, 2) - math.pow(0, 2)) / (2 * acc)
                    instant_speed += acc
                else:
                    acc_fuel_energies += vehicle_specific_power(instant_speed, slope, acc, m, A) / 3600
                    instant_speed += acc
            remaining_seconds = section_duration - acc_duration
        return acc_green_energies, [acc_fuel_energies, vehicle_specific_power(vf, slope, 0, m, A) * remaining_seconds / 3600], acc_green_energies + acc_fuel_energies

class HybridBusProblem(Problem):

    """
    Hybrid Bus Problem representation
    """

    def __init__(self, route: Route, bus: Bus, population_size: int = 100000, process_id: int = 0, representatives: list = []):
        super(HybridBusProblem, self).__init__()
        self.representatives = representatives
        self.process_id = process_id
        self.route = route
        self.bus = bus
        self.population_size = population_size

        self.number_of_sections = len(self.route.sections)

        count = 0
        for section in self.route.sections:
            if section.section_type == 1:
                count += section.section_type
            elif section.slope <= -0.02:
                count += 1

        self.number_of_ze_sections = count


        self.number_of_objectives = 2
        self.number_of_variables = self.number_of_sections - self.number_of_ze_sections
        self.number_of_constraints = 4

        self.initial_solution = True
        self.epochs = 1
        self.evaluations = 0

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ["Green Kms Travelled", "Emitted Gases"]

    def single_compute_green_zones_maximums(self):
        self.green_zones_max_emissions = [0.0, 0.0, 0.0, 0.0]

        for index, section in enumerate(self.route.sections):
            section_emissions = 0
            if self.route.sections[index].section_type == 0 and self.route.sections[index].slope > -0.02:
                if self.route.sections[index].green_zone > 0:
                    if self.route.sections[index].bus_stop == 1:
                        _, kW_h, _ = acceleration_section_power(0, self.route.sections[index].speed, 0.7, self.route.sections[index].slope,
                        self.route.sections[index].distance, self.route.sections[index].seconds, 0)
                    else:
                        kW_h = [vehicle_specific_power(self.route.sections[index].speed , self.route.sections[index].slope,
                        0, self.bus.weight, self.bus.frontal_section) * self.route.sections[index].seconds/3600]
                    for kwh in kW_h:
                        if kwh >= 0:
                            gasoline_gallon_equivalent = kwh / self.bus.fuel_engine_efficiency * 0.02635046113 # Conversion factor

                            section_emissions += gasoline_gallon_equivalent * 10.180 # Kgs of CO2 emissions

                    self.green_zones_max_emissions[self.route.sections[index].green_zone - 1] += section_emissions

        self.green_zones_max_emissions = np.array(self.green_zones_max_emissions, dtype=float)

    def total_compute_green_zones_maximums(self):
        for index, problem in enumerate(self.problems):
            if index != self.process_id:
                self.green_zones_max_emissions += problem.green_zones_max_emissions

        self.green_zones_max_emissions *= 0.5

        for index, problem in enumerate(self.problems):
            if index != self.process_id:
                problem.green_zones_max_emissions = self.green_zones_max_emissions
        print(self.green_zones_max_emissions)
    def evaluate_green_zones(self, solution):
        count = 0
        evaluation_array = []

        for section in self.route.sections:
            if section.section_type == 1:
                evaluation_array.append(1.0)
            else:
                # Si tiene una inclinación menor a -2% se hace siempre en eléctrico
                if section.slope > -0.02:
                    evaluation_array.append(int(solution.variables[count]))
                    count += 1
                else:
                    evaluation_array.append(1.0)

        green_zones_emissions = [0.0, 0.0, 0.0, 0.0]

        for index, section in enumerate(self.route.sections):
            section_emissions = 0
            if self.route.sections[index].section_type == 0 and self.route.sections[index].slope > -0.02:
                if self.route.sections[index].green_zone > 0 and evaluation_array[index] == 0:
                    if self.route.sections[index].bus_stop == 1:
                        _, kW_h, _ = acceleration_section_power(0, self.route.sections[index].speed, 0.7, self.route.sections[index].slope,
                        self.route.sections[index].distance, self.route.sections[index].seconds, 0)
                    else:
                        kW_h = [vehicle_specific_power(self.route.sections[index].speed , self.route.sections[index].slope,
                        0, self.bus.weight, self.bus.frontal_section) * self.route.sections[index].seconds/3600]
                    for kwh in kW_h:
                        if kwh >= 0:
                            gasoline_gallon_equivalent = kwh / self.bus.fuel_engine_efficiency * 0.02635046113 # Conversion factor

                            section_emissions += gasoline_gallon_equivalent * 10.180 # Kgs of CO2 emissions

                    if self.route.sections[index].green_zone > 0:
                        green_zones_emissions[self.route.sections[index].green_zone - 1] += section_emissions

        return green_zones_emissions

    def evaluate(self, solution: BinarySolution) -> BinarySolution:
        if self.evaluations == self.population_size:
            self.epochs += 1
            self.evaluations = 0

        self.evaluations += 1
        count = 0
        evaluation_array = []
        #print(solution.variables)
        for section in self.route.sections:
            if section.section_type == 1:
                evaluation_array.append(1.0)
            else:
                # Si tiene una inclinación menor a -2% se hace siempre en eléctrico
                if section.slope > -0.02:
                    evaluation_array.append(int(solution.variables[count]))
                    count += 1
                else:
                    evaluation_array.append(1.0)
        """ VSP Model application in order to obtain the objectives"""
        total_emissions = 0
        green_kms = 0
        remaining_charge = self.bus.charge
        remaining_charges = []
        green_zone_emissions = [0.0, 0.0, 0.0, 0.0]
        recharge = 0
        invalid = False
        for index,section in enumerate(evaluation_array):
            kW_h = 0
            fuel_kW_h = 0
            battery_kW_h = 0
            section_emissions = 0
            if section == 0:
                """
                    (VSP (kW) * Section Time (h) -> Energy Required to travel the section (kWh)
                """
                if self.route.sections[index].bus_stop == 1:
                    green_kWh, kW_h, recharge = acceleration_section_power(0, self.route.sections[index].speed, 0.7, self.route.sections[index].slope,
                    self.route.sections[index].distance, self.route.sections[index].seconds, section)

                    start_engine_battery = green_kWh / self.bus.electric_engine_efficiency
                    green_kms += 0.025
                    remaining_charge = decrease_battery_charge(remaining_charge, start_engine_battery, self.bus.charge)

                else:
                    kW_h = [vehicle_specific_power(self.route.sections[index].speed , self.route.sections[index].slope,
                    0, self.bus.weight, self.bus.frontal_section) * self.route.sections[index].seconds/3600]
                for kwh in kW_h:
                    if kwh < 0:
                        gasoline_gallon_equivalent = 0
                        remaining_charge = decrease_battery_charge(remaining_charge, kwh / self.bus.electric_engine_efficiency, self.bus.charge)
                    else:
                        gasoline_gallon_equivalent = kwh / self.bus.fuel_engine_efficiency * 0.02635046113 # Conversion factor

                    section_emissions += gasoline_gallon_equivalent * 10.180 # Kgs of CO2 emissions


                if  self.route.sections[index].green_zone > 0:
                    section_emissions *= 2


                total_emissions += section_emissions

            else:
                """
                    (VSP (kW) * Section Time (h) -> Energy Required to travel the section (kWh)
                """
                if self.route.sections[index].bus_stop == 1:
                    kW_h, _, recharge = acceleration_section_power(0, self.route.sections[index].speed, 0.7, self.route.sections[index].slope,
                    self.route.sections[index].distance, self.route.sections[index].seconds, section)
                else:

                    kW_h = vehicle_specific_power(self.route.sections[index].speed, self.route.sections[index].slope,
                    0, self.bus.weight, self.bus.frontal_section) * self.route.sections[index].seconds/3600


                section_battery = kW_h / self.bus.electric_engine_efficiency
                section_green_kms = self.route.sections[index].distance

                remaining_charge = decrease_battery_charge(remaining_charge, section_battery, self.bus.charge)
                green_kms += section_green_kms


            #print("{} : {} ".format(self.route.sections[index].bus_stop, section))
            #print("{} , {} , {}".format(kW_h, battery_kW_h, fuel_kW_h))
            #input('')

            if (index + 1) >= len(self.route.sections) or self.route.sections[index + 1].bus_stop == 1:
                recharge = recharge / self.bus.electric_engine_efficiency * 0.5
                if remaining_charge + recharge > self.bus.charge:
                    remaining_charge = self.bus.charge
                else:
                    remaining_charge += recharge
                recharge = 0
            remaining_charges.append(remaining_charge)

            if remaining_charge < 0:
                invalid = True

        # Penalizing invalid solutions
        if invalid:
            solution, green_kms, total_emissions = self.new_repair_solution(solution, remaining_charges, green_kms, total_emissions)
        else:
            green_kms *= -1

        green_zone_emissions = self.evaluate_green_zones(solution)
        self.partners = [None for x in range(len(self.representatives))]
        for index, solutions in enumerate(self.representatives):
            if index != self.process_id:
                repre = list(self.representatives)[index]

                sol = BinarySolution(number_of_variables=self.number_of_variables, number_of_objectives=2, number_of_constraints=4)
                sol.variables = [x for x in repre]

                other_green_kms, other_emissions, other_green_zone_emissions, sol = self.problems[index].old_evaluate(sol)

                green_zone_emissions = np.array(green_zone_emissions)
                aux = green_zone_emissions
                green_zone_emissions += np.array(other_green_zone_emissions)

                green_kms += other_green_kms
                total_emissions += other_emissions
                self.partners[index] = sol


        solution.objectives[0] = green_kms
        solution.objectives[1] = total_emissions
        solution.green_zone_emissions = green_zone_emissions


        solution = self.__evaluate_constraints(solution)

        return solution

    def __evaluate_constraints(self, solution: BinarySolution) -> None:
        constraints = [0.0, 0.0, 0.0, 0.0]

        green_zone_emissions = solution.green_zone_emissions

        constraints[0] =  self.green_zones_max_emissions[0] - green_zone_emissions[0]
        constraints[1] = self.green_zones_max_emissions[1] - green_zone_emissions[1]
        constraints[2] = self.green_zones_max_emissions[2] - green_zone_emissions[2]
        constraints[3] =  self.green_zones_max_emissions[3] - green_zone_emissions[3]

        #print(f'{green_zone_emissions} ## {self.green_zones_max_emissions}')
        solution.constraints = constraints

        return solution

    def old_evaluate(self, solution: BinarySolution) -> BinarySolution:
        count = 0
        evaluation_array = []

        for section in self.route.sections:
            if section.section_type == 1:
                evaluation_array.append(1.0)
            else:
                # Si tiene una inclinación menor a -2% se hace siempre en eléctrico
                if section.slope > -0.02:
                    evaluation_array.append(int(solution.variables[count]))
                    count += 1
                else:
                    evaluation_array.append(1.0)
        """ VSP Model application in order to obtain the objectives"""
        total_emissions = 0
        green_kms = 0
        remaining_charge = self.bus.charge
        remaining_charges = []
        recharge = 0
        invalid = False
        for index,section in enumerate(evaluation_array):
            kW_h = 0
            fuel_kW_h = 0
            battery_kW_h = 0
            section_emissions = 0
            if section == 0:
                """
                    (VSP (kW) * Section Time (h) -> Energy Required to travel the section (kWh)
                """
                if self.route.sections[index].bus_stop == 1:
                    green_kWh, kW_h, recharge = acceleration_section_power(0, self.route.sections[index].speed, 0.7, self.route.sections[index].slope,
                    self.route.sections[index].distance, self.route.sections[index].seconds, section)

                    start_engine_battery = green_kWh / self.bus.electric_engine_efficiency
                    green_kms += 0.025
                    remaining_charge = decrease_battery_charge(remaining_charge, start_engine_battery, self.bus.charge)

                else:
                    kW_h = [vehicle_specific_power(self.route.sections[index].speed , self.route.sections[index].slope,
                    0, self.bus.weight, self.bus.frontal_section) * self.route.sections[index].seconds/3600]
                for kwh in kW_h:
                    if kwh < 0:
                        gasoline_gallon_equivalent = 0
                        remaining_charge = decrease_battery_charge(remaining_charge, kwh / self.bus.electric_engine_efficiency, self.bus.charge)
                    else:
                        gasoline_gallon_equivalent = kwh / self.bus.fuel_engine_efficiency * 0.02635046113 # Conversion factor

                    section_emissions += gasoline_gallon_equivalent * 10.180 # Kgs of CO2 emissions

                if self.route.sections[index].green_zone > 0:
                    section_emissions *= 2
                total_emissions += section_emissions

            else:
                """
                    (VSP (kW) * Section Time (h) -> Energy Required to travel the section (kWh)
                """
                if self.route.sections[index].bus_stop == 1:
                    kW_h, _, recharge = acceleration_section_power(0, self.route.sections[index].speed, 0.7, self.route.sections[index].slope,
                    self.route.sections[index].distance, self.route.sections[index].seconds, section)
                else:

                    kW_h = vehicle_specific_power(self.route.sections[index].speed, self.route.sections[index].slope,
                    0, self.bus.weight, self.bus.frontal_section) * self.route.sections[index].seconds/3600


                section_battery = kW_h / self.bus.electric_engine_efficiency
                section_green_kms = self.route.sections[index].distance

                remaining_charge = decrease_battery_charge(remaining_charge, section_battery, self.bus.charge)
                green_kms += section_green_kms

            #print("{} : {} ".format(self.route.sections[index].bus_stop, section))
            #print("{} , {} , {}".format(kW_h, battery_kW_h, fuel_kW_h))
            #input('')
            if (index + 1) >= len(self.route.sections) or self.route.sections[index + 1].bus_stop == 1:
                recharge = recharge / self.bus.electric_engine_efficiency * 0.5
                if remaining_charge + recharge > self.bus.charge:
                    remaining_charge = self.bus.charge
                else:
                    remaining_charge += recharge
                recharge = 0
            remaining_charges.append(remaining_charge)

            if remaining_charge < 0:
                invalid = True

        # Penalizing invalid solutions
        if invalid:
            solution, green_kms, total_emissions = self.new_repair_solution(solution, remaining_charges, green_kms, total_emissions)
        else:
            green_kms *= -1

        return green_kms, total_emissions, self.evaluate_green_zones(solution), solution


    def repair_solution(self, solution, remaining_charges):
        full_solution = []
        count = 0
        for section in self.route.sections:
            if section.section_type == 1:
                full_solution.append(1)
            else:
                # Si tiene una inclinación menor a -2% se hace siempre en eléctrico
                if section.slope > -0.02:
                    full_solution.append(int(solution.variables[count]))
                    count += 1
                else:
                    full_solution.append(1)

        for i, _ in enumerate(self.route.sections):
            if self.route.sections[i].slope > -0.02 and self.route.sections[i].section_type == 0:
                if remaining_charges[i] < 0:
                    full_solution[i] = 0
                    total_emissions, green_kms, remaining_charges, _ = self.simple_evaluate(full_solution)


        #print("{}:{}".format(green_kms, total_emissions))
        slopes = [section.slope for section in self.route.sections]
        indexes = np.argsort(slopes)
        for index in indexes:
            if self.route.sections[index].slope > -0.02 and self.route.sections[index].section_type == 0:
                full_solution[index] = 1

                total_emissions, green_kms, remaining_charges, battery = self.simple_evaluate(full_solution)
                # Miro si remaining charge es negativo -> deshago el cambio
                negativo = False
                for v in remaining_charges:
                    if v < 0:
                        negativo = True
                if negativo:
                    full_solution[index] = 0
                    total_emissions, green_kms, remaining_charges, _ = self.simple_evaluate(full_solution)

        count = 0
        for i, section in enumerate(self.route.sections):
            if section.slope > -0.02 and section.section_type == 0:
                solution.variables[count] = True if full_solution[i] == 1 else False
                count += 1

        return solution, -1 * green_kms, total_emissions

    def new_repair_solution(self, solution, remaining_charges, green_kms, total_emissions):
        full_solution = []
        count = 0
        for section in self.route.sections:
            if section.section_type == 1:
                full_solution.append(1)
            else:
                # Si tiene una inclinación menor a -2% se hace siempre en eléctrico
                if section.slope > -0.02:
                    full_solution.append(int(solution.variables[count]))
                    count += 1
                else:
                    full_solution.append(1)

        for i, _ in enumerate(self.route.sections):
            j = i - 1
            while remaining_charges[i] < 0 and j>=0:
                if self.route.sections[j].slope > -0.02 and self.route.sections[j].section_type == 0:
                    remaining_charges = self.section_evaluation_sub(remaining_charges, j, full_solution[j])
                    full_solution[j] = 0
                j -= 1
        count = 0
        for i, section in enumerate(self.route.sections):
            if section.slope > -0.02 and section.section_type == 0:
                solution.variables[count] = True if full_solution[i] == 1 else False
                count += 1
        total_emissions, green_kms, remaining_charges, _ = self.simple_evaluate(full_solution)
        if (np.array(remaining_charges)<0).sum() > 0:
            return self.repair_solution(solution, remaining_charges)

        return solution, -1 * green_kms, total_emissions

    def section_evaluation_sub(self, remaining_charges, index, section):
        kW_h = 0
        section_battery = 0
        if section < 1:
            if self.route.sections[index].bus_stop == 0:
                if section > 0.50:
                    """
                    (VSP (kW) * Section Time (h) -> Energy Required to travel the section part with battery (kWh)
                    """
                    battery_kW_h = vehicle_specific_power(self.route.sections[index].speed, self.route.sections[index].slope,
                    0, self.bus.weight, self.bus.frontal_section) * (self.route.sections[index].seconds/3600)

                    section_battery = battery_kW_h / self.bus.electric_engine_efficiency

                    fuel_kW_h = vehicle_specific_power(self.route.sections[index].speed, self.route.sections[index].slope,
                    0, self.bus.weight, self.bus.frontal_section) * (self.route.sections[index].seconds/3600)

                    for kwh in [fuel_kW_h]:
                        if kwh < 0:
                            section_battery = decrease_battery_charge(section_battery, kwh / self.bus.electric_engine_efficiency, self.bus.charge)


            else:
                """
                    (VSP (kW) * Section Time (h) -> Energy Required to travel the section(kWh)
                """

                battery_kW_h, fuel_kW_h, _ = acceleration_section_power(0, self.route.sections[index].speed, 0.7, self.route.sections[index].slope,
                self.route.sections[index].distance, self.route.sections[index].seconds, section)

                section_battery = battery_kW_h / self.bus.electric_engine_efficiency

                '''for kwh in fuel_kW_h:
                    if kwh < 0:
                        section_battery = section_battery + kwh / self.bus.electric_engine_efficiency'''

                _, fuel_kW_h, _ = acceleration_section_power(0, self.route.sections[index].speed, 0.7, self.route.sections[index].slope,
                self.route.sections[index].distance, self.route.sections[index].seconds, 0)

                for kwh in fuel_kW_h:
                    if kwh < 0:
                        section_battery = decrease_battery_charge(section_battery, kwh / self.bus.electric_engine_efficiency, self.bus.charge)


        elif section == 1:
            """
                (VSP (kW) * Section Time (h) -> Energy Required to travel the section (kWh)
            """
            if self.route.sections[index].bus_stop == 1:
                kW_h, _, _ = acceleration_section_power(0, self.route.sections[index].speed, 0.7, self.route.sections[index].slope,
                self.route.sections[index].distance, self.route.sections[index].seconds, section)

                _, fuel_kW_h, _ = acceleration_section_power(0, self.route.sections[index].speed, 0.7, self.route.sections[index].slope,
                self.route.sections[index].distance, self.route.sections[index].seconds, 0)

                for kwh in fuel_kW_h:
                    if kwh < 0:
                        section_battery = decrease_battery_charge(section_battery, kwh / self.bus.electric_engine_efficiency, self.bus.charge)

            else:

                kW_h = vehicle_specific_power(self.route.sections[index].speed, self.route.sections[index].slope,
                0, self.bus.weight, self.bus.frontal_section) * self.route.sections[index].seconds/3600

                fuel_kW_h = vehicle_specific_power(self.route.sections[index].speed, self.route.sections[index].slope,
                0, self.bus.weight, self.bus.frontal_section) * (self.route.sections[index].seconds/3600)

                for kwh in [fuel_kW_h]:
                    if kwh < 0:
                        section_battery = decrease_battery_charge(section_battery, kwh / self.bus.electric_engine_efficiency, self.bus.charge)



            section_battery = kW_h / self.bus.electric_engine_efficiency

        cont = index
        while cont < len(remaining_charges):
            remaining_charges[cont] = self.bus.charge if remaining_charges[cont] + section_battery > self.bus.charge else remaining_charges[cont] + section_battery
            cont += 1

        return remaining_charges



    def simple_evaluate(self, sol):
        count = 0
        evaluation_array = []
        for section in self.route.sections:
            evaluation_array.append(sol[count])
            count += 1
        """ VSP Model application in order to obtain the objectives"""
        total_emissions = 0
        array_emissions = []
        green_kms = 0
        remaining_charge = self.bus.charge
        section_battery = 0
        batteries = []
        remaining_charges = []
        recharge = 0
        for index,section in enumerate(evaluation_array):
            kW_h = 0
            fuel_kW_h = 0
            battery_kW_h = 0
            section_emissions = 0
            if section == 0:
                """
                    (VSP (kW) * Section Time (h) -> Energy Required to travel the section (kWh)
                """
                if self.route.sections[index].bus_stop == 1:
                    green_kWh, kW_h, recharge = acceleration_section_power(0, self.route.sections[index].speed, 0.7, self.route.sections[index].slope,
                    self.route.sections[index].distance, self.route.sections[index].seconds, section)

                    start_engine_battery = green_kWh / self.bus.electric_engine_efficiency
                    green_kms += 0.025
                    remaining_charge = decrease_battery_charge(remaining_charge, start_engine_battery, self.bus.charge)
                else:

                    kW_h = [vehicle_specific_power(self.route.sections[index].speed , self.route.sections[index].slope,
                    0, self.bus.weight, self.bus.frontal_section) * self.route.sections[index].seconds/3600]
                for kwh in kW_h:
                    if kwh < 0:
                        gasoline_gallon_equivalent = 0
                        remaining_charge = decrease_battery_charge(remaining_charge, kwh / self.bus.electric_engine_efficiency, self.bus.charge)
                    else:
                        gasoline_gallon_equivalent = kwh / self.bus.fuel_engine_efficiency * 0.02635046113 # Conversion factor

                    section_emissions += gasoline_gallon_equivalent * 10.180 # Kgs of CO2 emissions

                if self.route.sections[index].green_zone > 0:
                    section_emissions *= 2
                total_emissions += section_emissions
                array_emissions.append(section_emissions)

            else:
                """
                    (VSP (kW) * Section Time (h) -> Energy Required to travel the section (kWh)
                """
                if self.route.sections[index].bus_stop == 1:
                    kW_h, _, recharge = acceleration_section_power(0, self.route.sections[index].speed, 0.7, self.route.sections[index].slope,
                    self.route.sections[index].distance, self.route.sections[index].seconds, section)
                else:

                    kW_h = vehicle_specific_power(self.route.sections[index].speed, self.route.sections[index].slope,
                    0, self.bus.weight, self.bus.frontal_section) * self.route.sections[index].seconds/3600


                section_battery = kW_h / self.bus.electric_engine_efficiency
                section_green_kms = self.route.sections[index].distance

                remaining_charge = decrease_battery_charge(remaining_charge, section_battery, self.bus.charge)
                green_kms += section_green_kms

            #print("{} : {} ".format(self.route.sections[index].bus_stop, section))
            #print("{} , {} , {}".format(kW_h, battery_kW_h, fuel_kW_h))
            #input('')

            if (index + 1) >= len(self.route.sections) or self.route.sections[index + 1].bus_stop == 1:
                recharge = recharge / self.bus.electric_engine_efficiency * 0.5
                if remaining_charge + recharge > self.bus.charge:
                    remaining_charge = self.bus.charge
                else:
                    remaining_charge += recharge
                recharge = 0
            batteries.append(section_battery)
            remaining_charges.append(remaining_charge)

        return total_emissions, green_kms, remaining_charges, batteries

    def heuristic_individual(self):
        # heurística
        # 0 - Inicializamos las zonas ZE obligatorias
        sections = self.route.sections
        sol = [section.section_type * 1 for section in sections]

        # 1- todas las cuestas abajo en eléctrico
        for index, section in enumerate(sections):
            if section.slope < 0:
                sol[index] = 1

        emissions, energyConsumption, remainingCharge, _ = self.simple_evaluate(sol)

        # 2- meter los tramos llanos mientras haya batería:

        a = [section.slope for section in sections]
        for i in range(0,len(a)):
            if a[i] == 0.0 and sol[i] == 0:
                # Lo meto
                sol[i] = 1
                emissions, energyConsumption, remainingCharge, _ = self.simple_evaluate(sol)
                # Miro si remaining charge es negativo en algún tramo -> deshago el cambio
                negativo = False
                for v in remainingCharge:
                    if v < 0:
                        negativo = True

                if negativo:
                    sol[i] = 0
                    emissions, energyConsumption, remainingCharge, _ = self.simple_evaluate(sol)

        # 3- Por orden de manor a mayor cuesta ascendente, ir añadiendo tramos
        indexes = np.argsort(a)
        for index in indexes:
            if a[index]>0 and sol[index] == 0:
                # Lo meto
                sol[index] = 1
                emissions, energyConsumption, remainingCharge, _ = self.simple_evaluate(sol)

                # Miro si remaining charge es negativo -> deshago el cambio
                negativo = False
                for v in remainingCharge:
                    if v < 0:
                        negativo = True
                if negativo:
                    sol[index] = 0
                    emissions, energyConsumption, remainingCharge, _ = self.simple_evaluate(sol)
        return sol, emissions, energyConsumption, remainingCharge

    def create_solution(self) -> BinarySolution:

        new_solution = BinarySolution(number_of_variables=self.number_of_variables, number_of_objectives=self.number_of_objectives, number_of_constraints=self.number_of_constraints)

        if self.initial_solution:
            sample_solution, emissions, green_kms, _ = self.heuristic_individual()

            count = 0
            for i, section in enumerate(self.route.sections):
                if section.slope > -0.02 and section.section_type == 0:
                    new_solution.variables[count] = bool(sample_solution[i])
                    count+=1
            self.initial_solution = False
        else:
            new_solution.variables = \
                [bool(random.randint(0, 1)) for _ in range(self.number_of_sections - self.number_of_ze_sections)]

        return new_solution

    def calculate_total_greenK(self):
        _, emissions, green_kms, _ = self.heuristic_individual()

        for problem in self.problems:
            sample_solution, aux_emissions, aux_green_kms, _ = problem.heuristic_individual()
            emissions += aux_emissions
            green_kms += aux_green_kms
            print(sample_solution)

        print(f"{green_kms}, {emissions}")
        #input()

    def get_name(self) -> str:
      return 'Hybrid Bus CoEv'

