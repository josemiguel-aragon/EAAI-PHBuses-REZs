import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from jmetal.util.solution import get_non_dominated_solutions
from jmetal.core.solution import BinarySolution


def read_solutions_sync():
    all_solutions = []
    with open('results_sync_emissions_greenkms.data', 'r') as file:
        for line in file.readlines():
            if "Solution" in line:
                all_solutions.append(BinarySolution(number_of_variables = 1, number_of_objectives = 2))
            if "Variables:" in line:
                variables = line.split(':')[1].replace('[', '').replace(']', '').split(',')
                variables = [eval(x) for x in variables]
                all_solutions[-1].variables[0] = variables
            if 'Green_kms' in line:
                objectives = line.split(':')[1]
                all_solutions[-1].objectives[0] = float(objectives)
            if 'Emissions:' in line:
                objectives = line.split(':')[1]
                all_solutions[-1].objectives[1] = float(objectives)
    return all_solutions

def read_solutions_async():
    all_solutions = []
    with open('results_async_emissions_greenkms.data', 'r') as file:
        for line in file.readlines():
            if "Solution" in line:
                all_solutions.append(BinarySolution(number_of_variables = 1, number_of_objectives = 2))
            if "Variables:" in line:
                variables = line.split(':')[1].replace('[', '').replace(']', '').split(',')
                variables = [eval(x) for x in variables]
                all_solutions[-1].variables[0] = variables
            if 'Green_kms' in line:
                objectives = line.split(':')[1]
                all_solutions[-1].objectives[0] = float(objectives)
            if 'Emissions:' in line:
                objectives = line.split(':')[1]
                all_solutions[-1].objectives[1] = float(objectives)
    return all_solutions

def plot_final_pareto_front():
    aux = read_solutions_sync()
    aux += read_solutions_async()

    final_front = get_non_dominated_solutions(aux)

    fig = plt.figure()
    plt.scatter(x = [x.objectives[0] * -1 for x in final_front], y = [x.objectives[1] for x in final_front], figure = fig, color = 'slategrey', alpha=0.5, s = mpl.rcParams['lines.markersize'] ** 1.75)
    plt.xlabel("Electric Range (Km)",fontsize=16)
    plt.ylabel("Pollutants (Kg CO2)",fontsize=16)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    fig.tight_layout(pad=1.0)
    plt.savefig("nondom_solutions_without_green_k.png", dpi=300)

    print(f"DIFF EMISSIONS : {(np.max([x.objectives[1] for x in final_front]) - np.min([x.objectives[1] for x in final_front]))}")
    print(f"DIFF GREEN_KMS : {(np.max([x.objectives[0] * -1 for x in final_front]) - np.min([x.objectives[0] * -1 for x in final_front]))}")


if __name__ == '__main__':
    plot_final_pareto_front()