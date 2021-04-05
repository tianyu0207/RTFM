import visdom
import numpy as np
import torch

class Visualizer(object):
    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        self.index = {}

    def plot_lines(self, name, y, **kwargs):
        '''
        self.plot('loss', 1.00)
        '''
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=str(name),
                      opts=dict(title=name),
                      update=None if x == 0 else 'append',
                      **kwargs
                      )
        self.index[name] = x + 1
    def disp_image(self, name, img):
        self.vis.image(img=img, win=name, opts=dict(title=name))
    def lines(self, name, line, X=None):
        if X is None:
            self.vis.line(Y=line, win=name)
        else:
            self.vis.line(X=X, Y=line, win=name)
    def scatter(self, name, data):
        self.vis.scatter(X=data, win=name)

def process_feat(feat, length):
    new_feat = np.zeros((length, feat.shape[1])).astype(np.float32)
    
    r = np.linspace(0, len(feat), length+1, dtype=np.int)
    for i in range(length):
        if r[i]!=r[i+1]:
            new_feat[i,:] = np.mean(feat[r[i]:r[i+1],:], 0)
        else:
            new_feat[i,:] = feat[r[i],:]
    return new_feat


def minmax_norm(act_map, min_val=None, max_val=None):
    if min_val is None or max_val is None:
        relu = torch.nn.ReLU()
        max_val = relu(torch.max(act_map, dim=0)[0])
        min_val = relu(torch.min(act_map, dim=0)[0])

    delta = max_val - min_val
    delta[delta <= 0] = 1
    ret = (act_map - min_val) / delta

    ret[ret > 1] = 1
    ret[ret < 0] = 0

    return ret


def modelsize(model, input, type_size=4):
    # check GPU utilisation
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * type_size / 1000 / 1000))

    input_ = input.clone()
    input_.requires_grad_(requires_grad=False)

    mods = list(model.modules())
    out_sizes = []

    for i in range(1, len(mods)):
        m = mods[i]
        if isinstance(m, nn.ReLU):
            if m.inplace:
                continue
        out = m(input_)
        out_sizes.append(np.array(out.size()))
        input_ = out

    total_nums = 0
    for i in range(len(out_sizes)):
        s = out_sizes[i]
        nums = np.prod(np.array(s))
        total_nums += nums


    print('Model {} : intermedite variables: {:3f} M (without backward)'
          .format(model._get_name(), total_nums * type_size / 1000 / 1000))
    print('Model {} : intermedite variables: {:3f} M (with backward)'
          .format(model._get_name(), total_nums * type_size*2 / 1000 / 1000))


def save_best_record(test_info, file_path):
    fo = open(file_path, "w")
    fo.write("epoch: {}\n".format(test_info["epoch"][-1]))
    fo.write(str(test_info["test_AUC"][-1]))
    fo.close()