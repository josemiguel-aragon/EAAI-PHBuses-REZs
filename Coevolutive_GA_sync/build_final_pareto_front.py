from jmetal.util.solution import get_non_dominated_solutions
from jmetal.core.solution import BinarySolution

from tqdm import tqdm

route_labeling = [
    'Line 7',
    'Line 22',
    'Line H10',
    'Line H12',
    'Line V15',
    'Line V17'
]

def get_final_front():
    all_solutions =  []
    for process_id in tqdm(range(6)):
        folder = f'generation_front_files_process_{process_id}/'
        
        green_kms = []
        emissions = []
        with open(f'{folder}FUN.4') as file:
            for line in file.readlines():
                objectives = line.split(' ')
                green_kms.append(float(objectives[0]))
                emissions.append(float(objectives[1]))
        
        solutions = []
        with open(f'{folder}VAR.4') as file:
            for line in file.readlines():
                solution = {'variables': [], 'partners': []}
                
                solution_and_partners = line.split('|')
                aux = solution_and_partners[0].replace('[', '').replace(']', '').split(',')
                solution['variables'] = [eval(x) for x in aux]
                
                for i in range(1, len(solution_and_partners)):
                    if 'None' not in solution_and_partners[i]:
                        aux = solution_and_partners[i].replace('[', '').replace(']', '').split(',')
                        solution['partners'].append([eval(x) for x in aux])
                    else:
                        aux = solution_and_partners[0].replace('[', '').replace(']', '').split(',')
                        solution['partners'].append([eval(x) for x in aux])
                        
                solutions.append(solution)
        
        for index,_ in enumerate(solutions):
            sol = BinarySolution(number_of_variables = 1, number_of_objectives = 2)
            
            sol.variables[0] = solutions[index]['variables']
            sol.objectives[0] = green_kms[index]
            sol.objectives[1] = emissions[index]
            sol.partners = solutions[index]['partners']
            sol.label = route_labeling[process_id]
        
            all_solutions.append(sol)

    final_front = get_non_dominated_solutions(all_solutions)
    
    print("Final Pareto Front:")
    for sol in final_front:
        print("\tSolution:")
        for index,_ in enumerate(route_labeling):
            print(f"\t\t{route_labeling[index]}")
            print(f"\t\tVariables: {sol.partners[index]}")
        print(f"\t\tObjectives: {sol.objectives}")
        
    
    with open('./results.data', 'w') as file:
        file.write("Final Pareto Front:")
        for sol in final_front:
            file.write("\n\tSolution:")
            for index,_ in enumerate(route_labeling):
                file.write(f"\n\t\t{route_labeling[index]}")
                file.write(f"\n\t\tVariables: {sol.partners[index]}")
            file.write(f"\n\t\tObjectives: {sol.objectives}")
            
        
        

if __name__ == '__main__':
    get_final_front()