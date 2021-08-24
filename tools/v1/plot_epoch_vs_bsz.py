import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import convergence
import argparse
import os
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot Convergence Speed')
    parser.add_argument('results', type=str, nargs='+', default=[],
                    help='experiment result')
    parser.add_argument('--max-accuracy', type=float, default=0.91,
                    help='maximum accuracy to report')
    parser.add_argument('--max-batch-size', type=int, default=256,
                    help='maximum batch size to report')
    parser.add_argument('--save-figure', type=str, default=None,
                    help='File in which to save figure.')
    parser.add_argument('--test-set', type=str, default='valid', choices=['valid', 'test'],
            help="Dataset used to evaluate convergence, choice of validation ('valid') or test ('test'). (default: valid).")
    args = parser.parse_args()

    bsz_seen = set({})
    iid = {
      "d-psgd": {
        0.8: {},
        0.85: {},
        0.90: {},
        0.91: {},
        0.92: {},
      },
      "sgp": {
        0.8: {},
        0.85: {},
        0.90: {},
        0.91: {},
        0.92: {},
      },
      "d2": {
        0.8: {},
        0.85: {},
        0.90: {},
        0.91: {},
        0.92: {},
      }
    }
    non_iid = {
      "d-psgd": {
        0.8: {},
        0.85: {},
        0.90: {},
        0.91: {},
        0.92: {},
      },
      "sgp": {
        0.8: {},
        0.85: {},
        0.90: {},
        0.91: {},
        0.92: {},
      },
      "d2": {
        0.8: {},
        0.85: {},
        0.90: {},
        0.91: {},
        0.92: {},
      }
    }

    for d in args.results:
        for result in os.listdir(d):
            with open(os.path.join(d, result, 'meta.json'), 'r') as meta_file:
                meta = json.load(meta_file)

            bsz = meta['batch_size']
            if bsz > args.max_batch_size:
                continue

            bsz_seen.add(bsz)

            conv = convergence.from_dir(os.path.join(d,result), test_set=args.test_set)

            for acc in conv['epochs']:
                if acc > args.max_accuracy:
                    continue
                dd = iid if meta['local_classes'] == 10 else non_iid
                alg = meta['dist_optimization']
                dd[alg][acc][bsz] = conv['epochs'][acc]

    print(bsz_seen)
    print(iid)
    print(non_iid)
    #sys.exit(0)

    bsz = sorted(list(bsz_seen))
    colors = {
      "iid": {
        "d-psgd": plt.cm.Blues(np.linspace(1.0, 0.1, len(iid['d-psgd'].keys()))),
        "sgp": plt.cm.Blues(np.linspace(0.7, 0.1, len(iid['sgp'].keys()))),
        "d2": plt.cm.Blues(np.linspace(0.5, 0.1, len(iid['d2'].keys())))
      }, 
      "non-iid": {
        "d-psgd": plt.cm.Reds(np.linspace(0.90, 0.1, len(non_iid['d-psgd'].keys()))),
        "sgp": plt.cm.Reds(np.linspace(0.7, 0.1, len(iid['sgp'].keys()))),
        "d2": plt.cm.Reds(np.linspace(0.5, 0.6, len(iid['d2'].keys())))
      }
    }

    x = np.arange(len(bsz))  # the label locations
    width = 0.4  # the width of the bars

    fig, ax = plt.subplots()

    accuracies = [ acc for acc in [0.90] ]
    accuracies.reverse()

    # To show all three at once
    algos = [
      #('d-psgd',-2.5*width-0.04, -1.5*width-0.04), 
      #('sgp', -0.5*width, 0.5*width),
      #('d2', 1.5*width+0.04, 2.5*width+0.04)
    ]

    # To show only one
    algos = [
      ('d-psgd', 0., 0.), 
      ('sgp', 0., 0.),
      ('d2', 0., 0.)
    ]

    def dd_list(loffset, roffset):
        return [
            (iid, 'iid', loffset), 
            (non_iid, 'non-iid', roffset)
        ]

    curves = []
    for alg,loffset,roffset in algos:
        for dd,label,offset in dd_list(loffset,roffset):
            for acc,i in zip(accuracies, range(0, len(accuracies))):
                ydata = [ dd[alg][acc][b] for b in bsz if b in dd[alg][acc].keys() 
                                      and type(dd[alg][acc][b]) == int 
                                      and dd[alg][acc][b] > 0 ]
                xdata = (x + offset)[:len(ydata)]
                curve = ax.plot(
                    xdata, 
                    ydata, 
                    label='{} {:2.0f}% ({})'.format(alg, acc*100, label), 
                    color=colors[label][alg][i],
                    marker='.',
                    markerfacecolor=colors[label][alg][i]
                )
                curves.append(curve)

    # Add some text for batch-sizes, title and custom x-axis tick labels, etc.
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')
    ax.tick_params(bottom=False, left=False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='#EEEEEE')
    ax.xaxis.grid(False)

    ax.set_ylabel('Epochs')

    if args.save_figure is None:
        ax.set_title('Epochs Needed to First Reach % Accuracy in Average on All Nodes')

    ax.set_xticks(x)
    ax.set_xticklabels(bsz)
    ax.set_xlabel('Batch Sizes')
    ax.set_ylim([0,100])
    ax.legend(loc='upper left')


    # """Attach a text label above each curve in *curves*, displaying its height."""
    for curve in curves:
        for p in curve:
            if 'sgp' in p.get_label():
                continue
            for x,y in zip(p.get_xdata(), p.get_ydata()):
                if y >= 101:
                    text = ">100"
                elif y >= 1: 
                    text = round(y, 1)
                else:
                    text = ''
                ax.text(
                  x - 0.1,
                  y + 0.3,
                  text,
                  horizontalalignment='right',
                  color=p.get_color(),
                  weight='bold'
                )

    fig.tight_layout()

    if args.save_figure is not None:
        plt.savefig(args.save_figure, transparent=True, bbox_inches='tight')
    
    plt.show()
