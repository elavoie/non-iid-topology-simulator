import sys
import setup.model.linear as linear
import torch

def create(params):
    name = params['model']['name']

    if name == 'linear':
        return linear.create(params)
    else:
        sys.stderr.write('Unsupported model name {}'.format(name))
        sys.exit(1)

def average(models, weights=None):
    with torch.no_grad():
        if weights == None:
            weights = [ float(1./len(models)) for _ in range(len(models)) ]
        center_model = models[0].copy()
        for p in center_model.parameters():
            p.mul_(0)
        for m, w in zip(models, weights):
            for c1, p1 in zip(center_model.parameters(), m.parameters()):
                c1.add_(w*p1)
        return center_model

