import numpy as np

def _crop_size(data, size):
    shape = data.shape
    ret = []
    for s in size:
        if s in [Ellipsis, 0, None]:
            ret.append(Ellipsis)
        else:
            ret.append(s)
    return ret

def random_slice(data, size):
    size = _crop_size(data, size)
    shape = data.shape
    
    sli = []
    for i in range(len(size)):
        if size[i] is Ellipsis:
            sli.append(Ellipsis)
        else:
            r = np.random.randint(shape[i]-size[i])
            sli.append(slice(r, size[i]+r))
    return sli

def random_crop(data, size):
    sli = random_slice(data, size)
    return data[sli]

def center_slice(data, size):
    size = _crop_size(data, size)
    shape = data.shape
    
    sli = []
    for i in range(len(size)):
        if size[i] is Ellipsis:
            sli.append(Ellipsis)
        else:
            c = int(round((shape[i]-size[i])/2.))
            sli.append(slice(c, size[i]+c))
    return sli

def center_crop(data, size):
    sli = center_slice(data, size)
    return data[sli]
