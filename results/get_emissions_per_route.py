import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
params = {'axes.labelsize': 28,
         'xtick.labelsize': 24,
         'ytick.labelsize': 24}
pylab.rcParams.update(params)
import pandas as pd
import numpy as np
import os
import matplotlib.colors as mcolors



from shapely.geometry import Point, LineString
from ReadRoute import read_route
from Bus import Bus
from shapely.geometry import Polygon, Point, LineString


pd.options.mode.chained_assignment = None

# Colores armoniosos para las REZs
rez_colors = {
    "REZ1": np.array(mcolors.to_rgb("red")),  # Lavanda grisáceo
    "REZ2": np.array(mcolors.to_rgb("yellow")),  # Azul apagado
    "REZ3": np.array(mcolors.to_rgb("grey")),  # Verde gris suave
    "REZ4": np.array(mcolors.to_rgb("#8F9189")),   # Gris con matiz cálido
    "WHITE": np.array(mcolors.to_rgb("white"))   # Gris con matiz cálido
}

REZ1 = ["D20", "D50", "H12", "H14", "H16", "V13", "V15", "V17", "V19", "V21", "V27", "6", "7", "19", "22", "24", "39", "47", "52", "54", "59", "120", "136"]

REZ2 = ["V5", "13", "121", "125", "150"]

REZ3 = ["D20", "H6", "H8", "V1", "V3", "V5", "6", "7", "33", "34", "52", "54", "59", "175"]

REZ4 = ["H4", "H8", "V31", "V33", "11", "60", "126", "133"]

if __name__ == '__main__':
    bus_routes = []
    route_label = []
    config_route_paths = os.listdir('../solution_to_analyze/')
    for index, path in enumerate(config_route_paths):
        if "green_k.csv" in path:
            bus_routes.append(path.split('.csv')[0].replace("_green_k", ""))
            route_label.append(bus_routes[-1].split("_")[1].strip())
    

    ordering = []
    for route in route_label:
        assigned_rezs = []
        if route in REZ1:
            assigned_rezs.append("REZ1")
        if route in REZ2:
            assigned_rezs.append("REZ2")
        if route in REZ3:
            assigned_rezs.append("REZ3")
        if route in REZ4:
            assigned_rezs.append("REZ4")
        
        if len(assigned_rezs) >= 1:
                if route in REZ3:
                    ordering.append("2REZ3")
                elif route in REZ1:
                    ordering.append("1REZ1")
                elif route in REZ2:
                    ordering.append("3REZ2")
                elif route in REZ4:
                    ordering.append("4REZ4")
        else:
            ordering.append("Z")       

    
    CSV_files = ["green_k", "solution_bestREZs"]
    route_label = np.array(route_label)
    bus_routes = np.array(bus_routes)
    
    ordering = np.array(ordering)
    indices = np.argsort(ordering)
    print(len(ordering))
    print(ordering[indices])
    
    route_label = route_label[indices]
    bus_route = bus_routes[indices]
    
    rez_assignments = []
    for route in route_label:
        assigned_rezs = []
        if route in REZ3:
            assigned_rezs.append("REZ3")
            
        rez_assignments.append(assigned_rezs)

    fig, ax1 = plt.subplots(figsize=(22,5))

    bar_colors = ["indianred", "mediumseagreen"]
    labels=["GreenK", "CCMOCell"]
    alphas = [0.4, 0.6]
    line_colors = ["k", "grey"]
    count=0
    indices = np.where(["REZ3" in assigned_rezs for assigned_rezs in rez_assignments])[0]
    if len(indices) > 0:
        start = indices[0] - 0.5
        end = indices[-1] + 0.5
        
        ax1.fill_between([start, end], 0, 25, color=rez_colors["REZ3"], alpha=0.25)  # Sombra más clara para toda la REZ
        midpoint = (start + end) / 2  # Posición en el eje x
        ax1.text(midpoint, 23, "REZ 3", ha="center", va="center", fontsize=24, color="black")
        
    for CSV_file in CSV_files:
        emissions_per_route = []
        green_kms_per_route = []
        for bus_route in bus_routes:
            dataset = pd.read_csv(f"../solution_to_analyze/{bus_route}_{CSV_file}.csv", sep=',', index_col="Unnamed: 0.1")
            emissions_per_route.append(dataset["CO2 emissions"].sum())
            green_kms_per_route.append(dataset["Green kms"].sum())

        

        


        ax1.bar([x for x in range(0,70)], green_kms_per_route, color=bar_colors[count], alpha=alphas[count], label=f"Electric range {labels[count]}")
        ax2 = ax1.twinx()
        ax2.plot(emissions_per_route, marker="o", color=line_colors[count], label=f"Emissions {labels[count]}")
        ax1.set_ylim([0, 25])
        ax2.set_ylim([0, 20])
        count+=1

        
    




    
    
    ax2.set_ylabel("CO2 Emissions (Kg)", rotation=270, labelpad=25)
    ax1.set_ylabel("Electric range (Km)")
    ax1.set_xticks([x for x in range(0,70)], route_label, rotation=90, ha="left")
    #ax1.grid(axis="y", linestyle="--", alpha=0.5)


    ax1.set_xlim(-0.5, len(bus_routes) - 0.5)
    plt.tight_layout()
    plt.savefig(f"emissions_kms_per_line_all.png", dpi=600)


    # Crear la leyenda desde la figura principal
    legend = fig.legend(
        loc="upper left",
        bbox_to_anchor=(0.1, 1),
        frameon=False
    )

    # Guardar la leyenda como PNG
    legend_fig = plt.figure(figsize=(3, 2))  # Nueva figura solo para la leyenda
    legend_ax = legend_fig.add_subplot(111)
    legend_ax.axis("off")  # Ocultar ejes

    # Agregar la leyenda a la nueva figura
    legend_fig.legend(
        ncols=4,
        handles=legend.legendHandles,
        labels=[t.get_text() for t in legend.get_texts()],
        loc="center",
        frameon=False,
    )
    
    

    # Guardar la imagen de la leyenda
    #legend_fig.savefig("legend_emissions_per_zone.png", dpi=300, bbox_inches="tight", transparent=True)






