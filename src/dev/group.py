class Group(object):
    def __init__(self, basic_info, init_states=None, aux=None):
        # basic informations
        self.name = basic_info['name']
        self.population = basic_info['pop']
        self.gdp_per_capita = basic_info['gdp-per-capita']
        
        # initial states
        self.S = None
        self.E = None
        self.I = None
        self.A = None
        self.R = None
        self.D = None
        self.Q = None
        
        
        
        # parameters
        ## TBD
        
        
    
    def set_initial_states(self, init_states):
        # initial states
        self.S = init_states['S']
        self.E = init_states['E']
        self.I = init_states['I']
        self.A = init_states['A']
        self.R = init_states['R']
        self.D = init_states['D']
        self.Q = init_states['Q']
    
    def set_auxiliary_states(self, aux):
        # auxiliary fields
        self.variant = aux['var']
        self.socdist = aux['socdist']
        self.flight = aux['flight']