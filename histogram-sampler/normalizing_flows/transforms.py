import numpy as np

class uniform:
    def __init__(self, hist: np.histogramdd) -> None:
        self.hist, self.bins = hist
        self.shape = self.hist.shape
        self.dimension = len(self.shape)
    
    def get_data(self) -> np.ndarray:
        data = []

        for idx in np.ndindex(*self.shape):
            # Number of data points in this bin
            count = int(self.hist[idx])
            if count == 0:
                continue

            # Generate all data points for this bin
            data_point = [
                np.random.uniform(self.bins[d][idx[d]], self.bins[d][idx[d] + 1], size=count)
                for d in range(self.dimension)
            ]
            bin_data = np.stack(data_point, axis=1)
            data.append(bin_data)

        return np.vstack(data) if data else np.empty((0, self.dimension))
    
    def get_dim(self) -> int:
        return self.dimension
