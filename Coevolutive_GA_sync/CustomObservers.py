import logging
import os
from pathlib import Path
from typing import List, TypeVar

import numpy as np
from tqdm import tqdm

from jmetal.core.observer import Observer
from jmetal.core.problem import DynamicProblem
from jmetal.core.quality_indicator import InvertedGenerationalDistance
from jmetal.lab.visualization import Plot, StreamingPlot
from jmetal.util.solution import print_function_values_to_file,print_variables_to_file
from jmetal.util.solution import get_non_dominated_solutions

S = TypeVar("S")

LOGGER = logging.getLogger("jmetal")

class NonDomWriteFrontToFileObserver(Observer):
    def __init__(self, output_directory: str) -> None:
        """Write function values of the front into files.
        :param output_directory: Output directory. Each front will be saved on a file `FUN.x`."""
        self.counter = 0
        self.directory = output_directory

        if Path(self.directory).is_dir():
            LOGGER.warning("Directory {} exists. Removing contents.".format(self.directory))
            for file in os.listdir(self.directory):
                os.remove("{0}/{1}".format(self.directory, file))
        else:
            LOGGER.warning("Directory {} does not exist. Creating it.".format(self.directory))
            Path(self.directory).mkdir(parents=True)

    def update(self, *args, **kwargs):
        problem = kwargs["PROBLEM"]
        solutions = kwargs["SELF_SOLUTIONS"]

        if solutions:
            if isinstance(problem, DynamicProblem):
                termination_criterion_is_met = kwargs.get("TERMINATION_CRITERIA_IS_MET", None)

                if termination_criterion_is_met:
                    print_function_values_to_file(solutions, "{}/FUN.{}".format(self.directory, self.counter))
                    print_variables_to_file(solutions, problem.partners, "{}/VAR.{}".format(self.directory, self.counter))
                    self.counter += 1
            else:
                print_function_values_to_file(solutions, "{}/FUN.{}".format(self.directory, self.counter))
                print_variables_to_file(solutions, problem.partners, "{}/VAR.{}".format(self.directory, self.counter))
                self.counter += 1


class NonDomPlotFrontToFileObserver(Observer):
    def __init__(self, output_directory: str, step: int = 100, **kwargs) -> None:
        """Plot and save Pareto front approximations into files.
        :param output_directory: Output directory.
        """
        self.directory = output_directory
        self.plot_front = Plot(title="Pareto front approximation", **kwargs)
        self.last_front = []
        self.fronts = []
        self.counter = 0
        self.step = step

        if Path(self.directory).is_dir():
            LOGGER.warning("Directory {} exists. Removing contents.".format(self.directory))
            for file in os.listdir(self.directory):
                os.remove("{0}/{1}".format(self.directory, file))
        else:
            LOGGER.warning("Directory {} does not exist. Creating it.".format(self.directory))
            Path(self.directory).mkdir(parents=True)

    def update(self, *args, **kwargs):
        problem = kwargs["PROBLEM"]
        solutions = get_non_dominated_solutions(kwargs["SOLUTIONS"])
        solutions = [x for x in solutions if x.objectives[1] < 10000]
        evaluations = kwargs["EVALUATIONS"]

        if solutions:
            if (evaluations % self.step) == 0:
                if isinstance(problem, DynamicProblem):
                    termination_criterion_is_met = kwargs.get("TERMINATION_CRITERIA_IS_MET", None)

                    if termination_criterion_is_met:
                        if self.counter > 0:
                            igd = InvertedGenerationalDistance(np.array([s.objectives for s in self.last_front]))
                            igd_value = igd.compute(np.array([s.objectives for s in solutions]))
                        else:
                            igd_value = 1

                        if igd_value > 0.005:
                            self.fronts += solutions
                            self.plot_front.plot(
                                [self.fronts],
                                label=problem.get_name(),
                                filename=f"{self.directory}/front-{evaluations}",
                            )
                        self.counter += 1
                        self.last_front = solutions
                else:
                    self.plot_front.plot(
                        [solutions],
                        label=f"{evaluations} evaluations",
                        filename=f"{self.directory}/front-{evaluations}",
                    )
                    self.counter += 1
