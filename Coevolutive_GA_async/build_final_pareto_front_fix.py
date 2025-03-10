
from jmetal.util.solution import get_non_dominated_solutions
from jmetal.core.solution import BinarySolution

from tqdm import tqdm
from os import chdir
from Route import Route
from ReadRoute import read_route
from Bus import Bus
from evaluate_solutions import evaluate

import matplotlib.pyplot as plt
import numpy as np
import os
import glob

route_labeling = [
    'Line 7',
    'Line 22',
    'Line H10',
    'Line H12',
    'Line V15',
    'Line V17'
]

aux = os.listdir('../output/')
config_route_paths = []
for x in aux:
    if ".csv" in x:
        config_route_paths.append(x)

route_labeling = config_route_paths

def get_final_front():
    max_emissions = [25.52288452, 11.21214709,  9.74776675,  5.39528461]#[25.55137699, 16.7781324,   9.74776675,  5.39528461]#[5.47495382*2, 2.45894825*2, 1.55743134*2]
    all_fronts = []
    for exec_i in range(1,5+1):
        chdir(f'../Coevolutive_GA_fix_{exec_i}/')
        
        all_solutions =  []
        for process_id in tqdm(range(69+1)):
            folder = f'generation_front_files_process_{process_id}/'
            
            green_kms = []
            emissions = []
            func_path = sorted(glob.glob(f'{folder}FUN.*'), key=os.path.getmtime)[-1]
            var_path = sorted(glob.glob(f'{folder}VAR.*'), key=os.path.getmtime)[-1]
            with open(func_path) as file:
                for line in file.readlines():
                    objectives = line.split(' ')
                    green_kms.append(float(objectives[0]))
                    emissions.append(float(objectives[1]))
            
            solutions = []
            with open(var_path) as file:
                for line in file.readlines():
                    solution = {'variables': [], 'partners': []}
                    
                    solution_and_partners = line.split('|')
                    aux = solution_and_partners[0].replace('[', '').replace(']', '').strip().split(' ')
                    solution['variables'] = [eval(x) for x in aux if x]
                    
                    for i in range(1, len(solution_and_partners)):
                        if 'None' not in solution_and_partners[i]:
                            aux = solution_and_partners[i].replace('[', '').replace(']', '').strip().split(' ')
                            solution['partners'].append([eval(x) for x in aux if x])
                        else:
                            aux = solution_and_partners[0].replace('[', '').replace(']', '').strip().split(' ')
                            solution['partners'].append([eval(x) for x in aux if x])
                            
                    solutions.append(solution)
            
            for index,_ in enumerate(solutions):
                sol = BinarySolution(number_of_variables = 1, number_of_objectives = 2)
                
                sol.variables[0] = solutions[index]['variables']
                sol.objectives[0] = green_kms[index]
                sol.objectives[1] = emissions[index]
                sol.partners = solutions[index]['partners']
                all_solutions.append(sol)
                #green_kms = 0
                #emissions = 0 
                #green_zone_emissions = np.zeros(3)
                '''for j,partner in enumerate(sol.partners):
                    route = read_route("../output/"+config_route_paths[j])
                    bus  = Bus(1, route)
                    
                    aux_green_kms, aux_emissions, aux_green_zone_emissions = evaluate(partner, route, bus)
                    green_kms += aux_green_kms
                    emissions += aux_emissions
                    green_zone_emissions += aux_green_zone_emissions
                
                supera = False
                for index,_ in enumerate(green_zone_emissions):
                    if green_zone_emissions[index] > max_emissions[index]:
                        supera = True
                        print("LIADA")
                if not supera:
                    sol.label = route_labeling[process_id]
                    sol.objectives[0] = green_kms
                    sol.objectives[1] = emissions
                    sol.green_zone_emissions = green_zone_emissions
                    all_solutions.append(sol)
                '''
        final_front = get_non_dominated_solutions(all_solutions)
    
        #print("Final Pareto Front:")
        #for sol in final_front:
        #    print("\tSolution:")
        #    for index,_ in enumerate(route_labeling):
        #        print(f"\t\t{route_labeling[index]}")
        #        print(f"\t\tVariables: {sol.partners[index]}")
        #    print(f"\t\tObjectives: {sol.objectives}")
            
        with open('./results.data', 'w') as file:
            file.write("Final Pareto Front:")
            for sol in final_front:
                file.write("\n\tSolution:")
                for index,_ in enumerate(route_labeling):
                    file.write(f"\n\t\t{route_labeling[index]}")
                    file.write(f"\n\t\tVariables: {sol.partners[index]}")
                file.write(f"\n\t\tObjectives: {sol.objectives}")
                #file.write(f"\n\t\tGreen Zone Emissions: {sol.green_zone_emissions}")
    
        
        all_fronts += final_front

    final_front = get_non_dominated_solutions(all_fronts)
    
    print("Final Pareto Front:")
    for sol in final_front:
        print("\tSolution:")
        for index,_ in enumerate(route_labeling):
            print(f"\t\t{route_labeling[index]}")
            print(f"\t\tVariables: {sol.partners[index]}")
        print(f"\t\tObjectives: {sol.objectives}")
        
    with open('./results_all_fix.data', 'w') as file:
        file.write("Final Pareto Front:")
        for sol in final_front:
            file.write("\n\tSolution:")
            for index,_ in enumerate(route_labeling):
                file.write(f"\n\t\t{route_labeling[index]}")
                file.write(f"\n\t\tVariables: {sol.partners[index]}")
            file.write(f"\n\t\tObjectives: {sol.objectives}")
        
if __name__ == '__main__':
    get_final_front()
