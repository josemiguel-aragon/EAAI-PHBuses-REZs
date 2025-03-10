import pandas as pd
from Route import Route
from Section import Section
import Constants as ct
import numpy as np

# ****Reads the route.csv and returns a route with this information****
def read_route(input_file):
    sections = []
    total_kms = 0
    green_kms = 0

    # Read input file
    data = pd.read_csv(input_file, index_col=0)
    
    for index,row in data.iterrows():
        speed = float(row["Avg Speed"])
        slope = float(row["Slope Angle"])
        section_type = int(row["Zone Type"])
        distance = float(row["Distance"]) / 1000
        duration = float(row["Time"])
        bus_stop = int(row["Bus Stop"])
        green_zone = int(row["Green Zone"])
        acceleration = speed / duration if bus_stop == 1 else 0

        sec = Section(index, speed, slope, section_type, distance, duration, acceleration, bus_stop, green_zone)
        sections.append(sec)

        total_kms += sec.distance
        if sec.section_type == 1:
            green_kms += sec.distance
    
    #print("Total Route kms: {}, ZE kms: {}".format(total_kms, green_kms))
    route = Route(1, sections)

    return route



