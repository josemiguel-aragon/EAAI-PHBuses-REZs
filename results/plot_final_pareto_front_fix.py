import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np

def plot_final_pareto_front():
    green_kms = []
    emissions = []

    green_kms_async = []
    emissions_async = []

    with open('./results_all_sync.data') as file:
        for line in file.readlines():
            if 'Objectives' in line:
                objectives = line.split(':')[1].replace('[', '').replace(']', '').split(',')
                green_kms.append(float(objectives[0]) * -1)
                emissions.append(float(objectives[1]))

    with open('./results_all_fix.data') as file:
        for line in file.readlines():
            if 'Objectives' in line:
                objectives = line.split(':')[1].replace('[', '').replace(']', '').split(',')
                green_kms_async.append(float(objectives[0]) * -1)
                emissions_async.append(float(objectives[1]))

    with open('./green_k.data') as file:
        for line in file.readlines():
            if 'Objectives' in line:
                objectives = line.split(':')[1].replace('[', '').replace(']', '').split(',')
                green_k_green_kms = (float(objectives[0]) * -1)
                green_k_emissions = (float(objectives[1]))


    fig = plt.figure()
    plt.scatter(x = green_k_green_kms, y = green_k_emissions, figure = fig, color = 'mediumseagreen', alpha=0.5, s = mpl.rcParams['lines.markersize'] ** 1.75)
    plt.scatter(x = green_kms, y = emissions, figure = fig, color = 'slategrey', alpha=0.5, s = mpl.rcParams['lines.markersize'] ** 1.75)
    plt.scatter(x = green_kms_async, y = emissions_async, figure = fig, color = 'indianred', alpha=0.5, s = mpl.rcParams['lines.markersize'] ** 1.75)

    plt.legend(['GreenK', 'CCMOCell', "CCMOCell Fix"])
    plt.xlabel("$\it{f_d}$", fontsize=16)
    plt.ylabel("$\it{f_e}$", fontsize=16)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.savefig("final_front_fix.jpg", dpi=300)

    fig = plt.figure()
    plt.scatter(x = green_kms, y = emissions, figure = fig, color = 'slategrey', alpha=0.5, s = mpl.rcParams['lines.markersize'] ** 1.75)
    plt.scatter(x = green_kms_async, y = emissions_async, figure = fig, color = 'indianred', alpha=0.5, s = mpl.rcParams['lines.markersize'] ** 1.75)
    plt.xlabel("$\it{f_d}$", fontsize=16)
    plt.ylabel("$\it{f_e}$", fontsize=16)
    plt.legend(['CCMOCell', "CCMOCell Fix"])
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.savefig("final_front_fix_without_green_k.jpg", dpi=300)

    maximum_allowed = [25.52288452, 11.21214709,  9.74776675,  5.39528461]
    print(f"IMPROVEMENT EMISSIONS SYNC - GREENK: {(green_k_emissions - np.min(emissions)) / green_k_emissions * 100}")
    print(f"IMPROVEMENT GREEN_KMS SYNC - GREENK: {(green_k_green_kms - np.max(green_kms)) / green_k_green_kms * 100}")



if __name__ == '__main__':
    plot_final_pareto_front()