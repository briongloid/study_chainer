import chainer

class DCGANDataset(chainer.dataset.DatasetMixin):
    
    def __init__(self, make_z, dataset):
        self.make_z = make_z
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)
    
    def get_example(self, i):
        return (self.make_z(i), self.dataset[i])