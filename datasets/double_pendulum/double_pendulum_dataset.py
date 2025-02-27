import numpy as np

class DoublePendulumDataset:

    def __init__(self, path, reshape=True):
        self.path = path
        self.data = np.load(path + "/double_pendulum/pendulum_videos.npy") / 255
        if reshape:
            self.data = np.reshape(self.data, (self.data.shape[0] * self.data.shape[1], self.data.shape[2], self.data.shape[3], self.data.shape[4]))
            self.data = self.data[:, :, :, :1]
        else:
            self.data = self.data[:, :, :, :, :1]
        self.num_signals = self.data.shape[0]

    def __len__(self):
        return self.num_signals
    
    def __getitem__(self, idx):
        return self.data[idx]



if __name__ == "__main__":
    dset = DoublePendulumDataset("./data")
    print(dset[0])