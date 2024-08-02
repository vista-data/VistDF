class LazySeries:  # this should extends Series similar to GeoSeries.
    def __init__(self, fn, params):
        self.fn = fn
        self.params = params

    def execute(self):
        executed_params = [p.execute() for p in self.params]
        for params in zip(*executed_params):
            yield self.fn(*params)
    
    def apply(self, fn):
        return LazySeries(fn, [self])