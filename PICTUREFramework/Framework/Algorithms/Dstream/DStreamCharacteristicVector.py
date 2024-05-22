class DStreamCharacteristicVector:
    # rappresenta GRID

    # questo corrisponde ad una posizione sulla griglia
    # questo è un microcluster (cluster)
    # però manca un collegamento diretto alle coordinate sulla griglia
    # quindi... vediamo come se la cava l'array grid
    def __init__(self,
                 density=0.0,
                 label='NO_CLASS',
                 status='NORMAL',
                 density_category=None,
                 last_update_time=-1,
                 last_sporadic_removal_time=-1,
                 last_marked_sporadic_time=-1,
                 category_changed_last_time=False,
                 label_changed_last_iteration=True):
        self.last_update_time = last_update_time
        self.last_sporadic_removal_time = last_sporadic_removal_time
        self.last_marked_sporadic_time = last_marked_sporadic_time
        self.density_category = density_category
        self.density = density
        self.label = label
        self.status = status
        self.category_changed_last_time = category_changed_last_time
        self.label_changed_last_iteration = label_changed_last_iteration
        self.samples = []

    def add_sample(self, datum):
        self.samples.append(datum)

    def update_density(self, decay_factor, current_time):
        self.density = 1.0 + self.density * decay_factor ** (current_time - self.last_update_time)
        self.last_update_time = current_time