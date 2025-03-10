class Section:
    def __init__(self, identity, speed, slope, section_type, distance, seconds, acceleration, bus_stop, green_zone):
        """
        Class representing a section of the bus route

        :param identity: identifies the section
        :param speed: average speed of the section in km/h
        :param slope: terrain grade of inclination
        :param section_type: type of the section
        :param distance: distance in kms of the section
        :param seconds: seconds needed to complete the section
        :param acceleration: acceleration of the section in m/s²
        :param bus_stop: determine if the sections starts on a bus stop
        :param green_zone: determine if the section pass through a green area
        """
        self.identity = identity
        self.speed = speed
        self.slope = slope
        self.section_type = section_type
        self.distance = distance
        self.seconds = seconds
        self.acceleration = acceleration
        self.bus_stop = bus_stop
        self.green_zone = green_zone
