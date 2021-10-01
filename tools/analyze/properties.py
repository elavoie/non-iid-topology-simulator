properties = [
  (['meta', 'seed'], int, None),
  (['dataset', 'name'], str, None),
  (['dataset', 'train-examples-per-class'], list, None),
  (['dataset', 'validation-examples-per-class'], list, None),
  (['nodes', 'name'], str, None),
  (['nodes', 'nb-nodes'], int, None),
  (['topology', 'name'], str, None),
  (['topology', 'interclique-topology'], str, None),
  (['algorithm', 'clique-gradient'], bool, None),
  (['algorithm', 'unbiased-gradient'], bool, None),
  (['model', 'name'], str, None),
]

def get(params, path):
    obj = params
    for x in path:
        if not x in obj.keys():
            return None
        else:
            obj = obj[x]
    return obj
