
import pandas as pd
import matplotlib.pyplot as plt
import os
import tqdm
import numpy as np

config_input_folder = ["./Coevolutive_GA", "./Coevolutive_GA_async"]

if __name__ == '__main__':
    for input_folder in config_input_folder:
        #scenario = input_folder.split("_")[-1].replace("/", "")
        #route = input_folder.split("_")[2]
        #algorithm = input_folder.split("_")[1]
        os.mkdir(input_folder + "/data/")

        for iteration in range(1,30):
            if os.path.isdir(input_folder + f"_{iteration}"):
                with open(input_folder + f"_{iteration}" +"/results.data") as file:
                    green_kms = []
                    emissions = []
                    solutions = []
                    times = []
                    for line in file.readlines():
                        if "Fitness" in line:
                            values = line.split(":")[1].replace("[", "").replace("]", "")
                            emissions.append(float(values.split(',')[1]))
                            green_kms.append(float(values.split(',')[0]))
                        if "Variables" in line:
                            sol = line.split(":")[-1].replace("[", "").replace("]", "").replace(" ", "").strip().split(',')
                            sol = [int(x.strip()) for x in sol]
                            solutions[-1] += sol
                        if "Solution" in line:
                            solutions.append([])

                with open(input_folder + f"_{iteration}" +"/elapsed_time.txt") as file:
                    for line in file.readlines():
                        if "Tiempo total:" in line:
                            times.append(float(line.split(":")[-1].strip().split(' ')[0].strip()[0]))

                pd.DataFrame(green_kms, emissions).to_csv(input_folder  + "data/FUN.{}.tsv".format(iteration),
                            header = False, sep = " ")

                with open(input_folder  + "data/VAR.{}.tsv".format(iteration), "w") as varfile:
                    for sol in solutions:
                        varfile.write(str(sol))
                        varfile.write("\n")

                with open(input_folder  + "data/TIME.{}.tsv".format(iteration), "w") as varfile:
                    for sol in times:
                        varfile.write("{}\n".format(sol))
