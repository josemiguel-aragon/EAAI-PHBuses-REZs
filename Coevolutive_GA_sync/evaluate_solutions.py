from Route import Route
from ReadRoute import read_route
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


def evaluate(solution, route, bus):
    count = 0
    evaluation_array = []

    for section in route.sections:
        if section.section_type == 1:
            evaluation_array.append(1.0)
        else:
            # Si tiene una inclinación menor a -2% se hace siempre en eléctrico
            if section.slope > -0.02:
                evaluation_array.append(int(solution[count]))
                count += 1
            else:
                evaluation_array.append(1.0)
    """ VSP Model application in order to obtain the objectives"""
    total_emissions = 0
    green_kms = 0
    remaining_charge = bus.charge
    remaining_charges = []
    green_zone_emissions = [0.0, 0.0, 0.0]
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
            if route.sections[index].bus_stop == 1:
                green_kWh, kW_h, recharge = acceleration_section_power(0, route.sections[index].speed, 0.7, route.sections[index].slope,
                route.sections[index].distance, route.sections[index].seconds, section)
                
                start_engine_battery = green_kWh / bus.electric_engine_efficiency
                green_kms += 0.025
                remaining_charge = decrease_battery_charge(remaining_charge, start_engine_battery, bus.charge)
                
            else:
                kW_h = [vehicle_specific_power(route.sections[index].speed , route.sections[index].slope,
                0, bus.weight, bus.frontal_section) * route.sections[index].seconds/3600]
            for kwh in kW_h:
                if kwh < 0:
                    gasoline_gallon_equivalent = 0
                    remaining_charge = decrease_battery_charge(remaining_charge, kwh / bus.electric_engine_efficiency, bus.charge)
                else:
                    gasoline_gallon_equivalent = kwh / bus.fuel_engine_efficiency * 0.02635046113 # Conversion factor
            
                section_emissions += gasoline_gallon_equivalent * 10.180 # Kgs of CO2 emissions

            if  route.sections[index].green_zone == 1:
                green_zone_emissions[0] += section_emissions
                section_emissions *= 2
            elif route.sections[index].green_zone == 2:
                green_zone_emissions[1] += section_emissions
                section_emissions *= 2
            elif route.sections[index].green_zone == 3:
                green_zone_emissions[2] += section_emissions
                section_emissions *= 2
                    
            
            total_emissions += section_emissions

        else:
            """
                (VSP (kW) * Section Time (h) -> Energy Required to travel the section (kWh)
            """
            if route.sections[index].bus_stop == 1:
                kW_h, _, recharge = acceleration_section_power(0, route.sections[index].speed, 0.7, route.sections[index].slope,
                route.sections[index].distance, route.sections[index].seconds, section)
            else:

                kW_h = vehicle_specific_power(route.sections[index].speed, route.sections[index].slope,
                0, bus.weight, bus.frontal_section) * route.sections[index].seconds/3600


            section_battery = kW_h / bus.electric_engine_efficiency
            section_green_kms = route.sections[index].distance

            remaining_charge = decrease_battery_charge(remaining_charge, section_battery, bus.charge)
            green_kms += section_green_kms
        

        #print("{} : {} ".format(route.sections[index].bus_stop, section))
        #print("{} , {} , {}".format(kW_h, battery_kW_h, fuel_kW_h))
        #input('')

        if (index + 1) >= len(route.sections) or route.sections[index + 1].bus_stop == 1:
            recharge = recharge / bus.electric_engine_efficiency * 0.5
            if remaining_charge + recharge > bus.charge:
                remaining_charge = bus.charge
            else:
                remaining_charge += recharge
            recharge = 0
        remaining_charges.append(remaining_charge)

        if remaining_charge < 0:
            invalid = True
    
    # Penalizing invalid solutions
    if invalid:
        print("ENTRADO")
        solution, green_kms, total_emissions = new_repair_solution(solution, remaining_charges, green_kms, total_emissions, route, bus)
    else:
        green_kms *= -1
    return green_kms, total_emissions, np.array(green_zone_emissions)


def repair_solution(solution, remaining_charges, route, bus):
    full_solution = []
    count = 0
    for section in route.sections:
        if section.section_type == 1:
            full_solution.append(1)
        else:
            # Si tiene una inclinación menor a -2% se hace siempre en eléctrico
            if section.slope > -0.02:
                full_solution.append(int(solution[count]))
                count += 1
            else:
                full_solution.append(1)
                        
    for i, _ in enumerate(route.sections):
        if route.sections[i].slope > -0.02 and route.sections[i].section_type == 0:
            if remaining_charges[i] < 0:
                full_solution[i] = 0
                total_emissions, green_kms, remaining_charges, _ = simple_evaluate(full_solution, route, bus)

    
    #print("{}:{}".format(green_kms, total_emissions))       
    slopes = [section.slope for section in route.sections]
    indexes = np.argsort(slopes)
    for index in indexes:
        if route.sections[index].slope > -0.02 and route.sections[index].section_type == 0:
            full_solution[index] = 1
            
            total_emissions, green_kms, remaining_charges, battery = simple_evaluate(full_solution, route, bus)
            # Miro si remaining charge es negativo -> deshago el cambio
            negativo = False
            for v in remaining_charges:
                if v < 0:
                    negativo = True
            if negativo:
                full_solution[index] = 0
                total_emissions, green_kms, remaining_charges, _ = simple_evaluate(full_solution, route, bus)

    count = 0
    for i, section in enumerate(route.sections):
        if section.slope > -0.02 and section.section_type == 0:
            solution[count] = True if full_solution[i] == 1 else False
            count += 1

    return solution, -1 * green_kms, total_emissions
    
def new_repair_solution(solution, remaining_charges, green_kms, total_emissions, route, bus):
    full_solution = []
    count = 0
    for section in route.sections:
        if section.section_type == 1:
            full_solution.append(1)
        else:
            # Si tiene una inclinación menor a -2% se hace siempre en eléctrico
            if section.slope > -0.02:
                full_solution.append(int(solution[count]))
                count += 1
            else:
                full_solution.append(1)
    
    for i, _ in enumerate(route.sections):
        j = i - 1
        while remaining_charges[i] < 0 and j>=0:
            if route.sections[j].slope > -0.02 and route.sections[j].section_type == 0:
                remaining_charges = section_evaluation_sub(remaining_charges, j, full_solution[j], route, bus)
                full_solution[j] = 0
            j -= 1
    
    count = 0
    for i, section in enumerate(route.sections):
        if section.slope > -0.02 and section.section_type == 0:
            solution[count] = True if full_solution[i] == 1 else False
            count += 1
            
    total_emissions, green_kms, remaining_charges, _ = simple_evaluate(full_solution, route, bus)
    if (np.array(remaining_charges)<0).sum() > 0:
        return repair_solution(solution, remaining_charges, route, bus)

            
    return solution, -1 * green_kms, total_emissions

def section_evaluation_sub(remaining_charges, index, section, route, bus):
    kW_h = 0
    section_battery = 0
    if section == 1:
        """
            (VSP (kW) * Section Time (h) -> Energy Required to travel the section (kWh)
        """
        if route.sections[index].bus_stop == 1:
            kW_h, _, _ = acceleration_section_power(0, route.sections[index].speed, 0.7, route.sections[index].slope,
            route.sections[index].distance, route.sections[index].seconds
            , section)

            green_kW_h, fuel_kW_h, _ = acceleration_section_power(0, route.sections[index].speed, 0.7, route.sections[index].slope,
            route.sections[index].distance, route.sections[index].seconds, 0)
            
            for kwh in fuel_kW_h:
                if kwh < 0:
                    section_battery = decrease_battery_charge(section_battery, kwh / bus.electric_engine_efficiency, bus.charge)

        else:
            green_kW_h = 0
            kW_h = vehicle_specific_power(route.sections[index].speed, route.sections[index].slope,
            0, bus.weight, bus.frontal_section) * route.sections[index].seconds/3600

            fuel_kW_h = vehicle_specific_power(route.sections[index].speed, route.sections[index].slope,
            0, bus.weight, bus.frontal_section) * (route.sections[index].seconds/3600)

            for kwh in [fuel_kW_h]:
                if kwh < 0:
                    section_battery = decrease_battery_charge(section_battery, kwh / bus.electric_engine_efficiency, bus.charge)

        section_battery += (kW_h)/ bus.electric_engine_efficiency
        section_battery = decrease_battery_charge(section_battery, green_kW_h / bus.electric_engine_efficiency, bus.charge)
        
    cont = index
    while cont < len(remaining_charges):
        remaining_charges[cont] = bus.charge if remaining_charges[cont] + section_battery > bus.charge else remaining_charges[cont] + section_battery 
        cont += 1

    return remaining_charges



def simple_evaluate(sol, route, bus):
    count = 0
    evaluation_array = []
    for section in route.sections:
        evaluation_array.append(sol[count])
        count += 1
    """ VSP Model application in order to obtain the objectives"""
    total_emissions = 0
    array_emissions = []
    green_kms = 0
    remaining_charge = bus.charge
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
            if route.sections[index].bus_stop == 1:
                green_kWh, kW_h, recharge = acceleration_section_power(0, route.sections[index].speed, 0.7, route.sections[index].slope,
                route.sections[index].distance, route.sections[index].seconds, section)
                
                start_engine_battery = green_kWh / bus.electric_engine_efficiency
                green_kms += 0.025
                remaining_charge = decrease_battery_charge(remaining_charge, start_engine_battery, bus.charge)
            else:

                kW_h = [vehicle_specific_power(route.sections[index].speed , route.sections[index].slope,
                0, bus.weight, bus.frontal_section) * route.sections[index].seconds/3600]
            for kwh in kW_h:
                if kwh < 0:
                    gasoline_gallon_equivalent = 0
                    remaining_charge = decrease_battery_charge(remaining_charge, kwh / bus.electric_engine_efficiency, bus.charge)
                else:
                    gasoline_gallon_equivalent = kwh / bus.fuel_engine_efficiency * 0.02635046113 # Conversion factor
            
                section_emissions += gasoline_gallon_equivalent * 10.180 # Kgs of CO2 emissions 

            if route.sections[index].green_zone > 0:
                section_emissions *= 2
            total_emissions += section_emissions
            array_emissions.append(section_emissions)
            
        else:
            """
                (VSP (kW) * Section Time (h) -> Energy Required to travel the section (kWh)
            """
            if route.sections[index].bus_stop == 1:
                kW_h, _, recharge = acceleration_section_power(0, route.sections[index].speed, 0.7, route.sections[index].slope,
                route.sections[index].distance, route.sections[index].seconds, section)
            else:

                kW_h = vehicle_specific_power(route.sections[index].speed, route.sections[index].slope,
                0, bus.weight, bus.frontal_section) * route.sections[index].seconds/3600


            section_battery = kW_h / bus.electric_engine_efficiency
            section_green_kms = route.sections[index].distance

            remaining_charge = decrease_battery_charge(remaining_charge, section_battery, bus.charge)
            green_kms += section_green_kms
        
        #print("{} : {} ".format(route.sections[index].bus_stop, section))
        #print("{} , {} , {}".format(kW_h, battery_kW_h, fuel_kW_h))
        #input('')
        
        if (index + 1) >= len(route.sections) or route.sections[index + 1].bus_stop == 1:
            recharge = recharge / bus.electric_engine_efficiency * 0.5
            if remaining_charge + recharge > bus.charge:
                remaining_charge = bus.charge
            else:
                remaining_charge += recharge
            recharge = 0
        batteries.append(section_battery)
        remaining_charges.append(remaining_charge)

    return total_emissions, green_kms, remaining_charges, batteries


if __name__ == '__main__':
    config_route_paths = [
    '../output/processed_7_parades_linia_Barcelona_conAlturas_green_zones.csv',
    '../output/processed_22_parades_linia_Barcelona_conAlturas_green_zones.csv',
    '../output/processed_H10_parades_linia_Barcelona_conAlturas_green_zones.csv',
    '../output/processed_H12_parades_linia_Barcelona_conAlturas_green_zones.csv',
    '../output/processed_V15_parades_linia_Barcelona_conAlturas_green_zones.csv',
    '../output/processed_V17_parades_linia_Barcelona_conAlturas_green_zones.csv'
    ]

    solutions = [[False, False, True, False, False, False, False, True, False, True, False, True, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, True, True, False, True, True, True, True, True, True, False, True, False, True, True, True, True, True, True, True, True, True, True, True, False, False, True, False, True] ,[False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, True, True, False, True, True, True, True, True, True, True] ,[True, True, True, False, True, False, False, False, False, True, True, True, True, True, True, True, True, True, False, True, True, True, True, False, False, False, False, True, True, True, True, True, True, False, True, True, True, False, False, False, False, False, False, False, False, False, True, True, True, True, True, False, False, False, True, False, False, False, False, False, True, True, True, True, True, True, False, True, True, False, True, False] ,[True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, True, False, True, True, True, True, False, True, False, True, True, True, True, True, False, False, False, True, True, False, False, False, False, True, False, False, True, False, True, True, False, False, False, False, False, False, False, False, False, False, False, False, True, True, True, False, True, False, True, True, True, True, True, True] ,[False, True, True, True, False, True, False, True, True, True, False, True, True, True, True, True, True, False, False, True, False, True, True, False, False, True, True, False, False, True, True, False, False, False, False, False, False, False, False, False, False, False, False, True] ,[True, False, False, True, False, True, True, False, True, True, False, True, False, False, False, False, False, False, True, True, False, False, False, False, True, False, False, False, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]]
    total_green_kms = 0
    total_emissions = 0 
    total_green_zone_emissions = np.zeros(3)
    for index,_ in enumerate(solutions):
        route = read_route(config_route_paths[index])
        bus  = Bus(1, route)
        
        green_kms, emissions, green_zone_emissions = evaluate(solutions[index], route, bus)
        
        total_green_kms += green_kms
        total_emissions += emissions 
        total_green_zone_emissions += green_zone_emissions
    
    print(f"Green_kms: {total_green_kms}")
    print(f"Emissions: {total_emissions}")
    print(f"Low-Emissions Zone: {total_green_zone_emissions}")
        
        
        
  