

def MetricLogger():
    def __init__(self, filename, **meta):
        self.meta = meta
        self.writer = jsonlines.open(filename, mode='w')

    def update_meta(self, **meta):
        self.meta = {**self.meta, **meta}
    
    def log_array(self, array, name, idx_name):
        pass

    def log(self, value, name, **meta):
        self._log(value=value, name=name, **{**self.meta, **meta})

    def _log(self, **row):
        writer.write(row, flush=False)

    def __del__(self):
        self.writer.close()