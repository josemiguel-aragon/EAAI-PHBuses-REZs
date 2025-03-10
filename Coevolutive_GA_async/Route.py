class Route:
    def __init__(self, identity, sections):
        """

        :param identity: identifies the route
        :param sections: an array of the sections of the route
        time, so there are n + 1 times, where n is the number of sections
        """
        self.identity = identity
        self.sections = sections
