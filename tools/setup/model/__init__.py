import sys
import setup.model.linear as linear

def create(params):
    name = params['model']['name']

    if name == 'linear':
        return linear.create(params)
    else:
        sys.stderr.write('Unsupported model name {}'.format(name))
        sys.exit(1)
