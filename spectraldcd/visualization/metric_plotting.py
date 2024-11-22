import numpy as np
import matplotlib.pyplot as plt
import pickle 
import time 
from matplotlib.colors import to_rgba, to_hex
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score
import spectraldcd.experiments.metrics as metrics
import os
import pickle

dir_path = os.path.dirname(os.path.realpath(__file__))
style_path = os.path.join(dir_path, 'style.mplstyle')

plt.style.use(style_path)


#plt.style.use('spectraldcd/visualization/style.mplstyle')

def plot_nmi_vs_T(algorithm, folder, T_arr, totalsims, fig=None, ax=None):
    if ax is None:
        ax = fig.add_subplot(111)

    median_nmi_values = []
    lower_percentile_values = []
    upper_percentile_values = []

    with open(os.path.join(folder, f'all_{algorithm}_labels_all_T.pkl'), 'rb') as f:
        all_labels_all_T = pickle.load(f)

    with open(os.path.join(folder, 'all_ground_truths_all_T.pkl'), 'rb') as f:
        true_labels_all_T = pickle.load(f)

    for i, T in enumerate(T_arr):
                # Initialize lists to store NMI values for each simulation
        nmi_values = np.zeros((totalsims, T))

        all_labels = all_labels_all_T[i]
        all_true_labels = true_labels_all_T[i]
        print(np.shape(all_labels))

        # Iterate over each simulation
        for sim in range(totalsims):
            pred_labels = all_labels[sim]
            true_labels = all_true_labels[sim]
            # Compute the NMI between the predicted labels and the ground truth labels
            print(np.shape(pred_labels), np.shape(true_labels))
            for t in range(len(pred_labels)):
                print(np.shape(pred_labels[t]), np.shape(true_labels[t]))
                nmi = normalized_mutual_info_score(true_labels[t], pred_labels[t])
                nmi_values[sim, t] = nmi

        summary = "median" # can be "mean" or "median"
        if summary == "mean":
            summary_over_time = np.mean(nmi_values, axis=1) # shape is (num sims,)
        elif summary == "median":
            summary_over_time = np.median(nmi_values, axis=1)
        else:
            raise ValueError("summary must be 'mean' or 'median'")
        # now compute median and percnetiles over all sims
        median_nmi = np.median(summary_over_time) # this one is always median (median over all sims)
        lower_percentile = np.percentile(summary_over_time, 25)
        upper_percentile = np.percentile(summary_over_time, 75)

        median_nmi_values.append(median_nmi)
        lower_percentile_values.append(lower_percentile)
        upper_percentile_values.append(upper_percentile)

    # Plot the mean NMI values with error bars
    if algorithm == 'point_tangent':
        ax.errorbar(T_arr, median_nmi_values, yerr=[np.array(median_nmi_values) - np.array(lower_percentile_values), np.array(upper_percentile_values) - np.array(median_nmi_values)], fmt='o-', label='Geodesic Spectral Clustering')
    elif algorithm == 'spectral':
        ax.errorbar(T_arr, median_nmi_values, yerr=[np.array(median_nmi_values) - np.array(lower_percentile_values), np.array(upper_percentile_values) - np.array(median_nmi_values)], fmt='o-', label='Static Spectral Clustering')
    else:
        ax.errorbar(T_arr, median_nmi_values, yerr=[np.array(median_nmi_values) - np.array(lower_percentile_values), np.array(upper_percentile_values) - np.array(median_nmi_values)], fmt='o-', label=algorithm.replace('_', ' ').title())
    #if algorithm == 'point_tangent':
    #    ax.set_label('Geodesic Spectral Clustering')
#
    #if algorithm == 'spectral': # special cases because i don't want to change their naming conventions elsewhere
    #    ax.set_label('Static Spectral Clustering')

    # Set the x-axis label
    ax.set_xlabel(r'Max $T$, $p_\text{switch}=0.1/(T/10)$')

    # Set the y-axis label
    ax.set_ylabel(f'Median (over simulations) of {summary} (over time) NMI')

    # Set the title of the plot
    ax.set_title(f'{summary.capitalize()} NMI vs Max $T$')

    # Show the legend
    #ax.legend().set_draggable(state=True)

    return ax



def plot_nmi_vs_pin(algorithm, folder, p_in_arr, totalsims, T, fig=None, ax=None, ami=False):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Initialize lists to store the mean and percentiles of NMI values for each algorithm
    median_nmi_values = []
    lower_percentile_values = []
    upper_percentile_values = []

    with open(os.path.join(folder, f'all_{algorithm}_labels_all_probs.pkl'), 'rb') as f:
        print(f'all_{algorithm}_labels_all_probs.pkl')
        all_labels_all_probs = pickle.load(f)

    with open(os.path.join(folder, 'all_ground_truths_all_probs.pkl'), 'rb') as f:
        true_labels_all_probs = pickle.load(f)

    # enumerate over different values of p_in
    for i, p_in in enumerate(p_in_arr):
        # Initialize array to store NMI values for each simulation
        nmi_values = np.zeros((totalsims, T))

        all_labels = all_labels_all_probs[i]
        all_true_labels = true_labels_all_probs[i]

        # Iterate over each simulation
        for sim in range(totalsims):
            pred_labels = all_labels[sim]
            true_labels = all_true_labels[sim]
            # Compute the NMI between the predicted labels and the ground truth labels
            for t in range(len(pred_labels)):
                nmi = normalized_mutual_info_score(true_labels[t], pred_labels[t]) if not ami else adjusted_mutual_info_score(true_labels[t], pred_labels[t])
                nmi_values[sim, t] = nmi

        # Compute median over time for each simulation
        summary_over_time = np.median(nmi_values, axis=1)
        
        # Compute median and percentiles over all simulations
        median_nmi = np.median(summary_over_time)
        lower_percentile = np.percentile(summary_over_time, 25)
        upper_percentile = np.percentile(summary_over_time, 75)

        median_nmi_values.append(median_nmi)
        lower_percentile_values.append(lower_percentile)
        upper_percentile_values.append(upper_percentile)

    # Plot parameters for each algorithm
    plot_params = {
        'point_tangent': {'label': 'G-NSC', 'color': 'blue', 'linestyle': 'solid', 'marker': 4},
        'mod_max': {'label': 'G-SMM', 'color': 'red', 'linestyle': 'dashed', 'marker': 5},
        'geodesic_bhc': {'label': 'G-BHC', 'color': 'mediumseagreen', 'linestyle': 'dotted', 'marker': 6},
        'spectral': {'label': 'S-NSC', 'color': 'cornflowerblue', 'linestyle': 'solid', 'marker': 4},
        'static_mod_max': {'label': 'S-SMM', 'color': 'lightcoral', 'linestyle': 'dashed', 'marker': 5},
        'static_bhc': {'label': 'S-BHC', 'color': 'springgreen', 'linestyle': 'dotted', 'marker': 6}
    }

    if algorithm in plot_params:
        params = plot_params[algorithm]
        
        # Plot the line with lower opacity
        line = ax.plot(p_in_arr, median_nmi_values, 
                linestyle=params['linestyle'],
                color=params['color'],
                alpha=0.3,  # Lower opacity for the line
                label=params['label'])
        
        # Plot markers with higher opacity
        ax.plot(p_in_arr, median_nmi_values, 
                linestyle='None',  # No line, only markers
                marker=params['marker'],
                color=params['color'],
                alpha=1.0,  # Full opacity for markers
                markersize=7)
        
        # Add error bands
        ax.fill_between(p_in_arr, lower_percentile_values, upper_percentile_values, 
                        color=params['color'], alpha=0.1)
    else:
        # Default plotting for unknown algorithms
        ax.plot(p_in_arr, median_nmi_values, 'o-', label=algorithm.replace('_', ' ').title())
        ax.fill_between(p_in_arr, lower_percentile_values, upper_percentile_values, alpha=0.2)

    # Set labels and title
    ax.set_xlabel('$p_\\text{in}$', labelpad=-10, x=1.02)  
    ax.set_ylabel('Median NMI Over Time', labelpad=-10) if not ami else ax.set_ylabel('Median AMI Over Time', labelpad=-10)
    
    # Adjust y-axis limits
    ymin, ymax = 0, 1  # Set fixed limits for NMI scores
    ymax_extended = ymax + 0.02  # Add 1% above 1 for extra space
    ax.set_ylim(ymin, ymax_extended)
    
    # Adjust y-axis ticks to show only 0 and 1
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_yticklabels(['0.0', '', '', '', '', '1.0'])

    # Add a horizontal grid
    #ax.grid(axis='y', linestyle='--', alpha=0.7)

    return fig, ax






def add_custom_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    
    # Define the desired order
    desired_order = ['G-NSC', 'G-SMM', 'G-BHC', 'S-NSC', 'S-SMM', 'S-BHC']
    
    # Reorder handles and labels
    ordered_handles = []
    ordered_labels = []
    for label in desired_order:
        if label in labels:
            index = labels.index(label)
            ordered_handles.append(handles[index])
            ordered_labels.append(label)
    
    # Create the legend with the custom order
    ax.legend(ordered_handles,  ordered_labels, columnspacing=0.4, fontsize=8, ncol=2, loc='lower right')

def add_custom_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    
    # Define the desired order
    desired_order = ['G-NSC', 'G-SMM', 'G-BHC', 'S-NSC', 'S-SMM', 'S-BHC']
    
    # Reorder handles and labels
    ordered_handles = []
    ordered_labels = []
    for label in desired_order:
        if label in labels:
            index = labels.index(label)
            ordered_handles.append(handles[index])
            ordered_labels.append(label)
    
    # Create the legend with the custom order
    ax.legend(ordered_handles, ordered_labels, fontsize=8, ncol=2, loc='lower right')




def lighten_color(color, amount=0.5):
    """
    Lightens the given color by mixing it with white.
    :param color: String or tuple representing the color.
    :param amount: How much to lighten by. Range 0 to 1, where 0 is no change and 1 is white.
    :return: Lightened color as a hex string.
    """
    try:
        c = np.array(to_rgba(color))
        white = np.array([1, 1, 1, 1])
        color = (1 - amount) * c[:3] + amount * white[:3]  # blend with white
        return to_hex(color)
    except Exception:
        return color  # return the original color if conversion failed

def compute_metric(metric_name, pred_labels, true_labels=None, adjacency_matrices=None, onmi_threshold=None):
    if metric_name == 'ari':
        if true_labels is None:
            raise ValueError("True labels required for ARI computation.")
        return metrics.compute_and_plot_aris(pred_labels, true_labels, use_matlab_version=False, plot=False)
    elif metric_name == 'nmi':
        if true_labels is None:
            raise ValueError("True labels required for NMI computation.")
        return metrics.compute_and_plot_nmis(pred_labels, true_labels, plot=False)
    
    elif metric_name == 'ami':
        print("metric: AMI")
        if true_labels is None:
            raise ValueError("True labels required for AMI computation.")
        return metrics.compute_and_plot_amis(pred_labels, true_labels, plot=False)
    elif metric_name == 'mod':
        if adjacency_matrices is None:
            raise ValueError("Adjacency matrices required for modularity computation.")
        if true_labels is None:
            return metrics.compute_and_plot_modularities(pred_labels, adjacency_matrices, plot=False)
        else:
            return metrics.compute_and_plot_modularities(pred_labels, adjacency_matrices, true_partitions=true_labels, plot=False) 
    elif metric_name == 'cond':
        if adjacency_matrices is None:
            raise ValueError("Adjacency matrices required for conductance computation.")
        return metrics.compute_and_plot_conductances(pred_labels, adjacency_matrices, plot=False)
    elif metric_name == 'onmi':
        if true_labels is None:
            raise ValueError("True labels required for ONMI computation.")
        if onmi_threshold is None:
            raise ValueError("ONMI threshold required for ONMI computation.")
        return metrics.compute_and_plot_overlapping_nmis(pred_labels, true_labels, threshold=onmi_threshold, plot=False)

    elif metric_name == 'oews':
        if true_labels is None:
            raise ValueError("True labels required for OEW computation.")
        
        if onmi_threshold is None:
            raise ValueError("(Threshold required for OEW computation.")
        return metrics.compute_and_plot_overlapping_elementwise_similarities(pred_labels, true_labels, plot=False)

    elif metric_name == 'hnmi':
        if true_labels is None:
            raise ValueError("True labels required for HNMI computation.")
        return metrics.compute_and_plot_hnmis(pred_labels, true_labels, plot=False)
    
    elif metric_name == 'hecs':
        if true_labels is None:
            raise ValueError("True labels required for HECS computation.")
        return metrics.compute_and_plot_hecs(pred_labels, true_labels, plot=False)
    
    else:
        raise ValueError(f"Unknown metric: {metric_name}")



def plot_with_error_bars_multiple_metrics(predicted_labels_all_list, true_labels_all, legend_labels, metrics, adj_matrices_all=None, colors=None, ribbons=False, long_data=False, output_file=None, onmi_threshold=None, figuresize=None):
    if colors is None:
        colors = ['royalblue', 'crimson' , 'purple','darkorange'  , 'darkred', 'darkblue']
        #colors = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', '#AA3377', '#BBBBBB', '#000000']
    num_metrics = len(metrics)


    if figuresize is None:
        fig, axes = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 3), squeeze=False)
    else:
        fig, axes = plt.subplots(1, num_metrics, figsize=figuresize, squeeze=False)
    
        
    num_algorithms = len(predicted_labels_all_list) + 1 # +1 for the true labels in case of modularity

    # Setup subplots
    axes = axes.flatten()

    if true_labels_all is None:
        true_labels_all = [None] * len(predicted_labels_all_list) # only modularity and conductance can be computed
        print("true labels not provided, only modularity and conductance may be computed")
        if 'mod' not in metrics and 'cond' not in metrics:
            raise ValueError("True labels must be provided for ARI and NMI computation.")

    for metric_index, metric in enumerate(metrics):
        print(f"Plotting metric: {metric}")

        true_mod_plotted_flag = False
        
        if metric == 'mod':
            all_true_modularities = np.zeros((S,T))
        for alg_index, predicted_labels_all in enumerate(predicted_labels_all_list):
            print(f"For algorithm: {legend_labels[alg_index]}")
            S = len(predicted_labels_all)
            T = len(predicted_labels_all[0])
            all_metrics = np.zeros((S, T))

            for sim in range(S):
                print(f"Computing metric for simulation {sim+1}...")
                pred_labels = predicted_labels_all[sim]
                true_labels = true_labels_all[sim]


                if metric in ['ari', 'nmi', 'onmi', 'hnmi', 'ami', 'oews', 'hecs']:
                    met_values = compute_metric(metric, pred_labels, true_labels, onmi_threshold=onmi_threshold)
                elif metric in ['mod', 'cond']:
                    adjacency_matrices = adj_matrices_all[sim]
                    met_values, true_modularities = compute_metric(metric, pred_labels, true_labels, adjacency_matrices=adjacency_matrices)

                all_metrics[sim, :] = met_values
                if metric == 'mod':
                    all_true_modularities[sim, :] = true_modularities

            median_metrics = np.median(all_metrics, axis=0)
            q1_metrics = np.percentile(all_metrics, 25, axis=0)
            q3_metrics = np.percentile(all_metrics, 75, axis=0)

            if metric == 'mod':
                median_mod = np.median(all_true_modularities, axis=0)
                q1_mod = np.percentile(all_true_modularities, 25, axis=0)
                q3_mod = np.percentile(all_true_modularities, 75, axis=0)

            # Plotting for each algorithm within the current metric subplot
            time_steps = np.arange(T)
            if not ribbons and not long_data:
                axes[metric_index].errorbar(time_steps, median_metrics, yerr=[median_metrics - q1_metrics, q3_metrics - median_metrics],
                                            fmt='-o', color=colors[alg_index % len(colors)],
                                            ecolor=lighten_color(colors[alg_index % len(colors)], amount=0.7),
                                            elinewidth=2, capsize=5, capthick=2, markersize=2,
                                            label=legend_labels[alg_index], linewidth=2)
                axes[metric_index].spines['top'].set_visible(False)
                axes[metric_index].spines['right'].set_visible(False)
                axes[metric_index].spines['left'].set_color('gray') 
                axes[metric_index].spines['bottom'].set_color('gray')
                if metric == 'mod' and true_mod_plotted_flag == False:
                    axes[metric_index].plot(time_steps, median_mod, '-o', color='black', markersize=2, label='True Modularity', linewidth=2)
                    axes[metric_index].fill_between(time_steps, q1_mod, q3_mod, color='black', alpha=0.5)
                    true_mod_plotted_flag = True   
            elif not long_data:
                color = colors[alg_index % len(colors)]
                time_steps = np.arange(T)
                axes[metric_index].plot(time_steps, median_metrics, '-o', color=color, markersize=2, label=legend_labels[alg_index], linewidth=2)
                axes[metric_index].fill_between(time_steps, q1_metrics, q3_metrics, color=color, alpha=0.5)
                axes[metric_index].spines['top'].set_visible(False)
                axes[metric_index].spines['right'].set_visible(False)
                axes[metric_index].spines['left'].set_color('gray') 
                axes[metric_index].spines['bottom'].set_color('gray')   
                if metric == 'mod' and true_mod_plotted_flag == False:
                    axes[metric_index].plot(time_steps, median_mod, '-o', color='black', markersize=2, label='True Modularity', linewidth=2)
                    axes[metric_index].fill_between(time_steps, q1_mod, q3_mod, color='black', alpha=0.5)
                    true_mod_plotted_flag = True
                  
            else:
                assert(long_data)
                print("long data")
                color = colors[alg_index % len(colors)]
                time_steps = np.arange(0, T, 1) 
                axes[metric_index].plot(time_steps, median_metrics[::1], '-o', color=color, markersize=0.5, 
                                        label=legend_labels[alg_index], alpha=0.5)
                axes[metric_index].fill_between(time_steps, q1_metrics[::1], q3_metrics[::1], color=color, alpha=0.3)
                axes[metric_index].spines['top'].set_visible(False)
                axes[metric_index].spines['right'].set_visible(False)
                axes[metric_index].spines['left'].set_color('gray') 
                axes[metric_index].spines['bottom'].set_color('gray')
                axes[metric_index].grid(False)
                axes[metric_index].set_xticks(time_steps[0::20])
                axes[metric_index].set_xticklabels(time_steps[0::20])
            
                # Add light gray masks for specified time ranges
                #axes[metric_index].axvspan(18, 32, facecolor='lightgray', alpha=0.3)
                #axes[metric_index].axvspan(72, 86, facecolor='lightgray', alpha=0.3)
            
                if metric == 'mod' and not true_mod_plotted_flag:
                    axes[metric_index].plot(time_steps, median_mod[::1], '-o', color='black', markersize=2, 
                                            label='True Modularity', linewidth=2, alpha=0.7)
                    axes[metric_index].fill_between(time_steps, q1_mod[::1], q3_mod[::1], color='black', alpha=0.3)
                    true_mod_plotted_flag = True
            plt.ylim(0, 1.02)  # Extends to 1.02 instead of 1.0
            if metric == 'nmi' or metric == 'onmi' or metric == 'hnmi' or metric=='ami' or metric=='oews' or metric=='hecs':
                ymin, ymax = 0, 1.0  # Keep ymax at 1.0 for labeling purposes

                # Set specific ticks
                ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]

                # Create labels for first and last tick
                labels = [f'{tick:.1f}' if i == 0 or i == len(ticks) - 1 else '' for i, tick in enumerate(ticks)]

                # Set y-ticks with all labels
                plt.yticks(ticks=ticks, labels=labels)

                # Adjust top spine to match the extended limit
                plt.gca().spines['top'].set_bounds(0, 1.02)
                plt.gca().spines['right'].set_bounds(0, 1.02)


        # Customize each subplot
        axes[metric_index].set_xlabel('$t_i$', loc='right')
        axes[metric_index].xaxis.set_label_coords(1.02, -0.025) 
        if metric == 'ari':
            #axes[metric_index].set_ylabel(r"$\frac{\text{RI}-\mathbb{E}[\text{RI}]}{1 - \mathbb{E}[\text{RI}]}, \text{ where }\text{RI}=\frac{\#\text{true positives} + \#\text{true negatives}}{{C(n,2)}}$", fontsize=16, fontweight='bold')
            #axes[metric_index].set_title(f'Adjusted Rand Index over Time, {S} Simulations', fontsize=20, fontweight='bold')
            axes[metric_index].set_ylabel('Adjusted Rand Index', fontweight='bold')
        elif metric == 'nmi':
            #axes[metric_index].set_ylabel(r'$\frac{2 \times \mathtt{mutual\_info}(\text{true}; \text{estimate})}{H(\text{true}) + H(\text{estimate})}$', fontsize=16, fontweight='bold')
            #axes[metric_index].set_title(f'Normalized Mutual Information over Time, {S} Simulations', fontsize=16, fontweight='bold')
            axes[metric_index].set_ylabel('NMI', labelpad=-15)
            
        elif metric == 'ami':
            axes[metric_index].set_ylabel('AMI', labelpad=-15)
            
        elif metric == 'mod':
            #axes[metric_index].set_ylabel(r'Q=$\sum_{r}(e_{r}-a_{r}^{2})$', fontsize=16, fontweight='bold')
            #axes[metric_index].set_title(f'Modularity over Time, {S} Simulations', fontsize=20, fontweight='bold')
            axes[metric_index].set_ylabel('Modularity', fontweight='bold')
        elif metric == 'cond':
            axes[metric_index].set_ylabel('Conductance', fontweight='bold')
            axes[metric_index].set_title(f'Conductance over Time, {S} Simulations')
        elif metric == 'onmi':
            axes[metric_index].set_ylabel('Overlapping NMI'  ,labelpad=-15)
            #axes[metric_index].set_title(f'Overlapping NMI over Time, {S} Simulations', fontsize=16, fontweight='bold')
        elif metric == 'oews':
            axes[metric_index].set_ylabel('E-cS', labelpad=-15)
            #axes[metric_index].set_title(f'Overlapping EMS over Time, {S} Simulations', fontsize=16, fontweight='bold')
        elif metric == 'hnmi':
            axes[metric_index].set_ylabel('Hierarchical NMI',  labelpad=-15)
            #axes[metric_index].set_title(f'Hierarchical NMI over Time, {S} Simulations', fontsize=16, fontweight='bold')    
        elif metric == 'hecs':
            axes[metric_index].set_ylabel('E-cS',  labelpad=-15)
            #axes[metric_index].set_title(f'Hierarchical ECS over Time, {S} Simulations', fontsize=16, fontweight='bold')

        axes[metric_index].tick_params(axis='x')
        axes[metric_index].tick_params(axis='y')
        axes[metric_index].set_xticks(time_steps[0::4])
        axes[metric_index].grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        # place legend in center of figure
        #axes[metric_index].legend(loc='center', bbox_to_anchor=(0.5, 0.65), shadow=False, ncol=1, fontsize=12)
        #axes[metric_index].legend(loc='lower right', shadow=False, ncol=2)
        #axes[metric_index].legend(loc='lower center')
        
        #axes[metric_index].legend(loc='lower center', ncol=1, fontsize=7, bbox_to_anchor=(0.5, -0.035))
        axes[metric_index].legend(loc='lower center', ncol=2, fontsize=8)

    plt.tight_layout()
    if output_file is not None:
        plt.savefig(output_file, transparent=True, bbox_inches='tight')
    plt.show()
    
    if figuresize is not None:
        figlegend = plt.figure(figsize=(5.5/3, 5.5/5))
        figlegend.legend(axes[metric_index].get_legend_handles_labels()[0], axes[metric_index].get_legend_handles_labels()[1])

    if output_file is not None:
        # split extension and folder from file name
        output_file_folder = os.path.dirname(output_file)
        output_file_name = os.path.basename(output_file)
        if figuresize is not None:
            plt.savefig(os.path.join(output_file_folder, "legend_"+output_file_name), transparent=True)
    
    if figuresize is not None: 
        figlegend.show()

if __name__ == "__main__":
    pass # see benchmarking_workbench