import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import convergence_helper as convergence
import argparse
import os
import json
import sys
import torch

def get_data(result, args):
    ydata = []
    conv = None
    if args.yaxis == 'training-loss':
        conv = convergence.training_loss_from_dir(result)
        conv = conv if args.class_index is None else conv['classes'][args.class_index]
        ydata = conv['avg']
    elif args.yaxis == 'running-training-loss':
        conv = convergence.running_training_loss_from_dir(result)
        conv = conv if args.class_index is None else conv['classes'][args.class_index]
        ydata = conv['avg']
    elif args.yaxis == 'training-accuracy':
        conv = convergence.training_accuracy(result)
        conv = conv if args.class_index is None else conv['classes'][args.class_index]
        ydata = [ acc*100. for acc in conv['avg'] ]
    elif args.yaxis == 'validation-accuracy':
        conv = convergence.from_dir(result, test_set='valid')
        conv = conv if args.class_index is None else conv['classes'][args.class_index]
        ydata = [ acc*100. for acc in conv['avg'] ]
    elif args.yaxis == 'test-accuracy':
        conv = convergence.from_dir(result, test_set='test')
        conv = conv if args.class_index is None else conv['classes'][args.class_index]
        ydata = [ acc*100. for acc in conv['avg'] ]
    elif args.yaxis == 'scattering' or args.yaxis == 'consensus-distance':
        with open(os.path.join(result, 'events', 'global.jsonlines'), 'r') as global_file:
            events = []
            for line in global_file:
                e = json.loads(line)
                events.append(e)
        ydata = [ e['distance_to_center']['global']['avg'] for e in events 
                  if e['type'] == 'model-scattering' or  e['type'] == 'consensus-distance' ]
    elif args.yaxis == 'center-shift' or args.yaxis == 'efficiency' or args.yaxis == 'average-distance-travelled':
        events = []
        with open(os.path.join(result, 'events', 'global.jsonlines'), 'r') as global_file:
            for line in global_file:
                e = json.loads(line)
                events.append(e)
        centers = [ e['distance_to_center']['global']['center'] for e in events if e['type'] == 'model-scattering' ]
        delta_avgs = [ sum(e['deltas'])/len(e['deltas']) for e in events if e['type'] == 'model-scattering' ]

        def distance(c1, c2):
            c1_weight = torch.tensor(c1[0])
            c1_bias = torch.tensor(c1[1])
            c2_weight = torch.tensor(c2[0])
            c2_bias = torch.tensor(c2[1])
            return torch.sqrt(torch.sum((c1_weight - c2_weight)**2) + torch.sum((c1_bias - c2_bias)**2))

        shift = [ distance(centers[i], centers[i+1]) for i in range(len(centers)-1) ]
        
        if args.yaxis == 'center-shift':
            ydata = shift
        elif args.yaxis == 'average-distance-travelled':
            ydata = delta_avgs
        elif args.yaxis == 'efficiency':
            ydata = [ s / d  for d,s in zip(delta_avgs, shift) ]
    
    if args.yaxis == 'training-loss' or \
       args.yaxis == 'running-training-loss' or \
       args.yaxis == 'center-shift' or \
       args.yaxis == 'scattering' or \
       args.yaxis == 'consensus-distance' or \
       args.yaxis == 'average-distance-travelled' or \
       args.yaxis == 'efficiency':
        xdata = range(1,1+len(ydata))
    else:
        xdata = conv['sampling_epochs']

    return xdata, ydata, conv

def get_curves(name, xdata, ydata, conv, marker='', linestyle='-'):
    curves = []
    if args.show_std and conv != None:
        # Standard-deviation curve
        curve = ax.plot(
            xdata, 
            [ acc*100. for acc in conv['std'] ], 
            label='{}'.format(name),
            marker=marker,
            linestyle=linestyle
        )
        curves.append(curve)
        return curves
    elif ydata != None:
        # Average curve
        curve = ax.plot(
            xdata, 
            ydata, 
            label='{}'.format(name),
            marker=marker,
            linestyle=linestyle
        )
        curves.append(curve)


    # Min curve
    if args.add_min_max and (args.yaxis == 'validation-accuracy' or args.yaxis == 'test-accuracy' or args.yaxis == 'training-accuracy') and conv != None:
        min = ax.plot(
            xdata,
            [ acc*100. for acc in conv['min'] ],
            color=curve[0].get_color(),
            linewidth=0.5,
            linestyle=linestyle
        )
    elif args.add_min_max and conv != None:
        min = ax.plot(
            xdata,
            conv['min'],
            color=curve[0].get_color(),
            linewidth=0.5,
            linestyle=linestyle
        )

    # Max curve
    if args.add_min_max and (args.yaxis == 'validation-accuracy' or args.yaxis == 'test-accuracy' or args.yaxis == 'training-accuracy') and conv != None:
        max = ax.plot(
            xdata,
            [ acc*100. for acc in conv['max'] ],
            color=curve[0].get_color(),
            linewidth=0.5,
            linestyle=linestyle
        )
    elif args.add_min_max and conv != None:
        max = ax.plot(
            xdata,
            conv['max'],
            color=curve[0].get_color(),
            linewidth=0.5,
            linestyle=linestyle
        )

    # Standard Deviation curves
    if args.add_std and (args.yaxis == 'validation-accuracy' or args.yaxis == 'test-accuracy' or args.yaxis == 'training-accuracy') and conv != None:
        top = ax.plot(
            xdata,
            [ (avg+std)*100. for avg,std in zip(conv['avg'], conv['std']) ],
            color=curve[0].get_color(),
            linewidth=0.5,
            linestyle='--'
        )
        bottom = ax.plot(
            xdata,
            [ (avg-std)*100. for avg,std in zip(conv['avg'], conv['std']) ],
            color=curve[0].get_color(),
            linewidth=0.5,
            linestyle='--'
        )
    elif args.add_std and conv != None:
        top = ax.plot(
            xdata,
            [ (avg+std)*100. for avg,std in zip(conv['avg'], conv['std']) ],
            color=curve[0].get_color(),
            linewidth=0.5,
            linestyle='--'
        )
        bottom = ax.plot(
            xdata,
            [ (avg-std)*100. for avg,std in zip(conv['avg'], conv['std']) ],
            color=curve[0].get_color(),
            linewidth=0.5,
            linestyle='--'
        )
    return curves


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot Convergence Speed in Epochs')
    parser.add_argument('results', type=str, nargs='+', default=[],
                    help='experiment result')
    parser.add_argument('--save-figure', type=str, default=None,
                    help='File in which to save figure.')
    parser.add_argument('--yaxis', type=str, default='validation-accuracy', choices=['validation-accuracy', 'test-accuracy', 'training-accuracy', 'training-loss', 'running-training-loss', 'scattering', 'center-shift', 'average-distance-travelled', 'efficiency', 'consensus-distance'],
                    help='Metric to use on the yaxis. Computes average for all the nodes. (default: validation-accuracy)')
    parser.add_argument('--ymin', type=float, default=0.,
                    help='Minimum value on the y axis')
    parser.add_argument('--ymax', type=float, default=None,
                    help='Maximum value on the y axis (data dependent).')
    parser.add_argument('--labels', type=str, nargs='+', default=[],
                    help='Labels that will appear in the legend')
    parser.add_argument('--add-min-max', action='store_const', const=True, default=False,
                    help='add min and max curves')
    parser.add_argument('--add-std', action='store_const', const=True, default=False,
                    help='Add standard-deviation curves')
    parser.add_argument('--class-index', type=int, default=None,
                    help='Plot accuracy for specific class, instead of global average (default: False).')
    parser.add_argument('--show-std', action='store_const', const=True, default=False,
                    help='Show standard deviation of accuracy, instead of average (default: False).')
    parser.add_argument('--legend', type=str, default='best',
                    help='Position of legend (default: best).')
    parser.add_argument('--merge-min-max', type=str, nargs='+', default=[],
                    help='Merge the following list of experiments. Min and max are the min and max of all experiments. ')
    parser.add_argument('--no-legend', action='store_const', const=True, default=False,
                    help='Do not display legend (default: False).')
    parser.add_argument('--font-size', type=int, default=16,
                    help='Font size (default: 16).')
    parser.add_argument('--markers', type=str, nargs='+', default=[],
                    help='Markers used for each curve')
    parser.add_argument('--linestyles', type=str, nargs='+', default=[],
                    help='Linestyles used for each curve')

    args = parser.parse_args()
    print(args)

    matplotlib.rc('font', size=args.font_size)

    results = {}
    for result in args.results + args.merge_min_max:
        with open(os.path.join(result, 'params.json'), 'r') as meta_file:
            meta = json.load(meta_file)
        results[result] = { "meta": meta, "average": [], "minimum": [], "maximum": [] }

#    props = {}
#    for result in results:
#        meta = results[result]["meta"]
#        for p in meta:
#            if p not in props:
#                props[p] = set({})
#            props[p] = props[p].union(set({str(meta[p])}))
#
#    for superfluous in ['network_interface', 'log', 'learning_rate_list', 'batch_size_list', 'results_directory', 'averaging_neighbourhood', 'cliques']:
#        if superfluous in props:
#            del props[superfluous]        
#
#    same = [ (p,list(props[p])[0]) for p in props if len(props[p]) == 1 ]
#    different = [ (p,props[p]) for p in props if len(props[p]) > 1 ]
#
#    print('Identical parameters')
#    print('--------------------')
#    max_len_p = max([len(p) for (p,_) in same])
#    for (p,v) in same:
#        print(('{:' + str(max_len_p) + '} {}').format(p, v))
#    print()
#
#    print('Differing parameters')
#    print('--------------------')
#
    experiment_names = [ os.path.split(result)[1] if result[-1] != '/' else os.path.split(result[:-1])[1] for result in results ]
#    if len(different) > 0:
#        max_len_p = [ max([ len(name) for name in experiment_names ]) ] + [ max([ len(i) for i in s ] + [len(p)]) for (p,s) in different ]
#        f_str = " ".join([ "{:" + str(l) + "}" for l in max_len_p ])
#        print(f_str.format(*tuple(['experiment'] + [ p for (p,_) in different ])))
#        for name,result in zip(experiment_names, results):
#            print(f_str.format(*tuple([name] + [ str(results[result]["meta"][p]) if p in results[result]['meta'] else ' ' for (p,_) in different  ])))
#    else:
#        print('None')
#    print('')

    fig, ax = plt.subplots()
    curves = []

    # Remove merged results
    merged = {}
    if len(args.merge_min_max) > 0:
        for result in args.merge_min_max:
            merged[result] = results[result]
            results.pop(result)
        experiment_names = [ os.path.split(result)[1] if result[-1] != '/' else os.path.split(result[:-1])[1] for result in results ]
        experiment_names.append('merged')
    
    if len(args.labels) == 0:
        labels = experiment_names
    elif len(args.labels) < len(experiment_names):
        print('Insufficient number of labels')
        sys.exit(1)
    else:
        labels = args.labels

    if len(args.markers) == 0:
        markers = [ '' for _ in experiment_names ]
    elif len(args.markers) < len(experiment_names):
        print('Insufficient number of markers')
        sys.exit(1)
    else:
        markers = args.markers

    if len(args.linestyles) == 0:
        linestyles = [ '-' for _ in experiment_names ]
    elif len(args.linestyles) < len(experiment_names):
        print('Insufficient number of linestyles')
        sys.exit(1)
    else:
        linestyles = args.linestyles

    exps = [ k for k in results.keys() ] + [ [merged] ]

    for name,marker,linestyle,result in zip(labels, markers,linestyles,exps):
        if type(result) is not list:
            xdata, ydata, conv = get_data(result, args)
            curves.extend(get_curves(name, xdata, ydata, conv, marker=marker, linestyle=linestyle))
        elif len(result) > 0:
            data = [ get_data(result, args) for result in merged ]
            if data[0][2] == None:
                print('Yaxis option {} is incompatible with --merged-min-max'.format(args.yaxis))
                sys.exit(1)
            if not all(map(lambda d: len(d[0]) == len(data[0][0]), data)):
                print('Unmergeable results, results have different number of data points')
                sys.exit(1)
            conv = {
                "min": data[0][2]["min"],
                "max": data[0][2]["max"],
                "std": [ 0 for x in data[0][2]["std"] ],
                "avg": data[0][2]["avg"]
            }
            for i in range(len(data[0][0])):
                for d in data[1:]:
                    conv["min"][i] = min(conv["min"][i], d[2]["min"][i])
                    conv["max"][i] = max(conv["max"][i], d[2]["max"][i])
                    conv["avg"][i] += d[2]["avg"][i]
                conv["avg"][i] /= len(data)
                conv["avg"][i] *= 100. # average is reported in %
            curves.extend(get_curves(name, xdata, conv["avg"], conv, marker=marker, linestyle=linestyle))

    # Add some text for batch-sizes, title and custom x-axis tick labels, etc.
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')
    ax.tick_params(bottom=False, left=False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='#EEEEEE')
    ax.xaxis.grid(False)

    if args.yaxis == 'training-loss':
        ylabel = 'Training Loss' if args.show_std is not True else 'Std of Training Loss'
    elif args.yaxis == 'running-training-loss':
        ylabel = 'Training Loss (Running)' if args.show_std is not True else 'Std of Training Loss (Running)'
    elif args.yaxis == 'training-accuracy':
        ylabel = 'Training Accuracy' if args.show_std is not True else 'Std of Training Accuracy (%)'
    elif args.yaxis == 'validation-accuracy':
        ylabel = 'Validation Accuracy (%)' if args.show_std is not True else 'Std of Validation Accuracy (%)'
    elif args.yaxis == 'test-accuracy':
        ylabel = 'Test Accuracy (%)' if args.show_std is not True else 'Std of Test Accuracy (%)'
    elif args.yaxis == 'scattering' or args.yaxis == 'consensus-distance':
        ylabel = 'Consensus Distance'
    elif args.yaxis == 'center-shift':
        ylabel = 'Center Shift'
    elif args.yaxis == 'average-distance-travelled':
        ylabel = 'Average Distance Travelled'
    elif args.yaxis == 'efficiency':
        ylabel = 'Efficiency'

    ax.set_ylim(bottom=args.ymin)

    if args.ymax is not None:
        ax.set_ylim(top=args.ymax)

    ax.set_ylabel(ylabel)
    ax.set_xlabel('Epochs')

    if not args.no_legend:
        ax.legend(loc=args.legend)


    # """Attach a text label above each curve in *curves*, displaying its height."""
    if False: 
        for curve in curves:
            for p in curve:
                for x,y in zip(p.get_xdata(), p.get_ydata()):
                    ax.text(
                      x - 0.1,
                      y + (0.006 * 0.1),
                      y,
                      horizontalalignment='right',
                      color=p.get_color(),
                      weight='bold'
                    )

    fig.tight_layout()

    if args.save_figure is not None:
        plt.savefig(args.save_figure, transparent=True, bbox_inches='tight')
    
    plt.show()
