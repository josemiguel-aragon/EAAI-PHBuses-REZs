class Bus:
    def __init__(self, identity, route, charge = 8.9, weight = 19500, frontal_section = 7 , fuel_engine_efficiency = 0.6, electric_engine_efficiency = 0.85):
        """

        :param identity: identifies the bus
        :param route: the route assigned to the bus
        :param charge: the total available charge
        :param weight: the total weight of the bus
        :param fuel_engine_efficiency: efficiency of the fuel engine
        :param electric_engine_efficiency: efficiency of the electric engine
        """
        self.identity = identity
        self.route = route
        self.charge = charge
        self.weight = weight
        self.frontal_section = frontal_section
        self.fuel_engine_efficiency = fuel_engine_efficiency
        self.electric_engine_efficiency = electric_engine_efficiency