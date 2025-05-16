#!/usr/bin/env python3
import json
import numpy as np
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import seaborn as sns
import os
import math
import pandas as pd

# Constants
ENHANCEMENT_NAMES = {
    'enhancement1': 'Uncertainty Fusion',
    'enhancement2': 'TensorCP Workflow',
    'enhancement3': 'Multi-Height Attention'
}

COLOR_PALETTE = {
    'baseline': '#404040',
    'enhancement1': '#008080',
    'enhancement2': '#0066CC',
    'enhancement3': '#9933CC',
    'text': '#333333',
    'grid': '#DDDDDD'
}

def load_json(file_path):
    """Robust JSON file loading with validation."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("Invalid JSON structure")
            return data
    except Exception as e:
        raise RuntimeError(f"Error loading {file_path}: {str(e)}")

def calculate_percentage_change(new_value, old_value):
    """Safe percentage change calculation."""
    try:
        if old_value == 0:
            return 0.0 if new_value == 0 else float('inf')
        return ((new_value - old_value) / abs(old_value)) * 100
    except:
        return float('nan')

def get_latex_color(value):
    """Enhanced LaTeX color formatting."""
    if isinstance(value, str):
        return value
    if np.isnan(value):
        return "\\textcolor{gray}{N/A}"
    if value > 0:
        return f"\\textcolor{{teal}}{{\\textbf{{+{value:.1f}\\%}}}}"
    elif value < 0:
        return f"\\textcolor{{red}}{{\\textbf{{{value:.1f}\\%}}}}"
    return "\\textcolor{gray}{0.0\\%}"

def format_metric_value(value, metric_type='ap'):
    """Consistent value formatting across metrics."""
    if np.isnan(value):
        return "N/A"
    if metric_type == 'ap':
        return f"{value:.4f}"
    elif metric_type == 'error':
        return f"{value:.4f}"
    return str(value)

def setup_plot_style():
    """Comprehensive plot styling configuration with fallbacks."""
    try:
        # Try newest seaborn style first
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        try:
            # Try older seaborn style
            plt.style.use('seaborn-whitegrid')
        except:
            # Fallback to basic matplotlib style
            plt.style.use('ggplot')
    
    # Apply our customizations regardless of which style loaded
    mpl.rcParams.update({
        'figure.figsize': (8, 5),
        'font.family': 'serif',
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.dpi': 300,
        'grid.color': COLOR_PALETTE['grid'],
        'text.color': COLOR_PALETTE['text'],
        'axes.grid': True,
        'axes.edgecolor': '0.8',
        'axes.linewidth': 0.5
    })
    return COLOR_PALETTE
def create_output_dirs(output_dir):
    """Create output directory structure with validation."""
    output_path = Path(output_dir)
    figures_dir = output_path / "figures"
    tables_dir = output_path / "tables"
    
    for directory in [figures_dir, tables_dir]:
        directory.mkdir(parents=True, exist_ok=True)
        if not directory.is_dir():
            raise RuntimeError(f"Failed to create directory: {directory}")
    
    return {
        'figures': figures_dir,
        'tables': tables_dir
    }

def create_top_classes_chart(baseline_data, enhancements_data, output_dirs, colors, top_n=5):
    """Comprehensive top classes visualization."""
    # Data collection and validation
    all_classes = set(baseline_data.get("mean_dist_aps", {}).keys())
    for enh_data in enhancements_data.values():
        all_classes.update(enh_data.get("mean_dist_aps", {}).keys())
    
    valid_classes = [
        cls for cls in all_classes
        if (baseline_data.get("mean_dist_aps", {}).get(cls, 0)) > 0 or
        any(enh.get("mean_dist_aps", {}).get(cls, 0) > 0
        for enh in enhancements_data.values())
    ]
    
    if not valid_classes:
        raise ValueError("No valid classes found for comparison")
    
    valid_classes.sort(
        key=lambda x: max(
            enh.get("mean_dist_aps", {}).get(x, 0)
            for enh in enhancements_data.values()
        ),
        reverse=True
    )
    top_classes = valid_classes[:top_n]
    
    # Plot setup
    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(top_classes))
    width = 0.15
    
    # Baseline plot
    baseline_values = [
        baseline_data.get("mean_dist_aps", {}).get(cls, 0)
        for cls in top_classes
    ]
    ax.bar(
        x - width*1.5, baseline_values, width,
        label='Baseline', color=colors['baseline'],
        edgecolor='white', linewidth=0.5
    )
    
    # Enhancements plots
    for i, (enh_name, enh_data) in enumerate(enhancements_data.items()):
        enh_values = [
            enh_data.get("mean_dist_aps", {}).get(cls, 0)
            for cls in top_classes
        ]
        ax.bar(
            x - width*(0.5 - i), enh_values, width,
            label=ENHANCEMENT_NAMES.get(enh_name, enh_name),
            color=colors[enh_name],
            edgecolor='white', linewidth=0.5
        )
    
    # Plot formatting
    ax.set_ylabel('Average Precision (AP)', fontweight='bold')
    ax.set_title(
        f'Top {top_n} Classes by AP Across All Models',
        pad=20, fontweight='bold'
    )
    ax.set_xticks(x)
    ax.set_xticklabels(top_classes, rotation=45, ha='right')
    ax.legend(
        loc='upper right',
        bbox_to_anchor=(1, 1),
        ncol=2,
        framealpha=1
    )
    
    # Grid and borders
    ax.grid(True, axis='y', linestyle=':', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Save figure
    fig.tight_layout()
    fig_path = output_dirs['figures'] / "top_classes_ap.png"
    plt.savefig(fig_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return fig_path.name
def create_distance_threshold_chart(baseline_data, enhancements_data, class_name, output_dirs, colors):
    """Detailed distance threshold analysis."""
    distances = ["0.5", "1.0", "2.0", "4.0"]
    
    # Data collection
    baseline_values = []
    enh_values = {name: [] for name in enhancements_data}
    
    for dist in distances:
        baseline_values.append(
            baseline_data.get("label_aps", {}).get(class_name, {}).get(dist, 0)
        )
        for enh_name, enh_data in enhancements_data.items():
            enh_values[enh_name].append(
                enh_data.get("label_aps", {}).get(class_name, {}).get(dist, 0)
            )
    
    # Plot setup
    fig, ax = plt.subplots(figsize=(10, 6))
    x_values = [float(d) for d in distances]
    
    # Baseline plot
    ax.plot(
        x_values, baseline_values, 's--',
        linewidth=2, markersize=8,
        label='Baseline', color=colors['baseline']
    )
    
    # Enhancements plots
    markers = ['o', '^', 'D']
    for i, (enh_name, values) in enumerate(enh_values.items()):
        ax.plot(
            x_values, values, f'{markers[i]}-',
            linewidth=2, markersize=8,
            label=ENHANCEMENT_NAMES.get(enh_name, enh_name),
            color=colors[enh_name]
        )
    
    # Plot formatting
    ax.set_xlabel('Distance Threshold (meters)', fontweight='bold')
    ax.set_ylabel('Average Precision (AP)', fontweight='bold')
    ax.set_title(
        f'Distance Threshold Analysis: {class_name}',
        pad=20, fontweight='bold'
    )
    ax.legend(loc='best', framealpha=1)
    
    # Grid and borders
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Save figure
    fig.tight_layout()
    fig_path = output_dirs['figures'] / f"distance_threshold_{class_name.lower().replace(' ', '_')}.png"
    plt.savefig(fig_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return fig_path.name

def generate_latex_comparison_table(baseline_data, enhancements_data, output_dirs):
    """Comprehensive LaTeX table generation."""
    # Prepare metrics data structure
    metrics = {
        'Overall': {
            'Mean AP': {'baseline': baseline_data.get('mean_ap', float('nan'))},
            'ND Score': {'baseline': baseline_data.get('nd_score', float('nan'))},
            'Eval Time (s)': {'baseline': baseline_data.get('eval_time', float('nan'))}
        },
        'Classes': {},
        'Errors': {}
    }
    
    # Add enhancement metrics
    for enh_name, enh_data in enhancements_data.items():
        metrics['Overall']['Mean AP'][enh_name] = enh_data.get('mean_ap', float('nan'))
        metrics['Overall']['ND Score'][enh_name] = enh_data.get('nd_score', float('nan'))
        metrics['Overall']['Eval Time (s)'][enh_name] = enh_data.get('eval_time', float('nan'))
    
    # Class-specific metrics
    all_classes = set(baseline_data.get("mean_dist_aps", {}).keys())
    for enh_data in enhancements_data.values():
        all_classes.update(enh_data.get("mean_dist_aps", {}).keys())
    
    top_classes = sorted(
        all_classes,
        key=lambda x: max(
            enh.get("mean_dist_aps", {}).get(x, 0)
            for enh in enhancements_data.values()
        ),
        reverse=True
    )[:5]
    
    for cls in top_classes:
        metrics['Classes'][cls] = {
            'baseline': baseline_data.get("mean_dist_aps", {}).get(cls, 0)
        }
        for enh_name, enh_data in enhancements_data.items():
            metrics['Classes'][cls][enh_name] = enh_data.get("mean_dist_aps", {}).get(cls, 0)
    
    # Error metrics
    error_types = list(baseline_data.get("tp_errors", {}).keys())
    for err_type in error_types:
        readable_name = ' '.join([word.capitalize() for word in err_type.split('_')])
        metrics['Errors'][readable_name] = {
            'baseline': baseline_data.get("tp_errors", {}).get(err_type, float('nan'))
        }
        for enh_name, enh_data in enhancements_data.items():
            metrics['Errors'][readable_name][enh_name] = enh_data.get("tp_errors", {}).get(err_type, float('nan'))
    
    # Generate LaTeX table
    latex_content = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\caption{Comprehensive Performance Comparison Across All Models}",
        "\\label{tab:full_comparison}",
        "\\footnotesize",
        "\\renewcommand{\\arraystretch}{1.2}",
        "\\rowcolors{2}{gray!10}{white}",
        "\\begin{tabularx}{\\textwidth}{l" + "X" * (len(enhancements_data) + 1) + "}",
        "\\toprule",
        "\\rowcolor{gray!30}",
        "\\textbf{Metric} & \\textbf{Baseline} " +
        "".join(f"& \\textbf{{{ENHANCEMENT_NAMES.get(name, name)}}} "
               for name in enhancements_data) + "\\\\",
        "\\midrule"
    ]
    
    # Add overall metrics
    latex_content.append("\\multicolumn{" + str(len(enhancements_data)+2) + "}{l}{\\textbf{Overall Metrics}} \\\\")
    for metric in ['Mean AP', 'ND Score', 'Eval Time (s)']:
        row = [metric]
        row.append(format_metric_value(metrics['Overall'][metric]['baseline']))
        for enh_name in enhancements_data:
            row.append(format_metric_value(metrics['Overall'][metric][enh_name]))
        latex_content.append(" & ".join(row) + " \\\\")
    
    # Add class metrics
    latex_content.append("\\midrule")
    latex_content.append("\\multicolumn{" + str(len(enhancements_data)+2) + "}{l}{\\textbf{Class-Specific AP}} \\\\")
    for cls, values in metrics['Classes'].items():
        row = [cls]
        row.append(format_metric_value(values['baseline']))
        for enh_name in enhancements_data:
            row.append(format_metric_value(values[enh_name]))
        latex_content.append(" & ".join(row) + " \\\\")
    
    # Add error metrics
    latex_content.append("\\midrule")
    latex_content.append("\\multicolumn{" + str(len(enhancements_data)+2) + "}{l}{\\textbf{Error Metrics (Lower is Better)}} \\\\")
    for err_name, values in metrics['Errors'].items():
        row = [err_name]
        row.append(format_metric_value(values['baseline'], 'error'))
        for enh_name in enhancements_data:
            row.append(format_metric_value(values[enh_name], 'error'))
        latex_content.append(" & ".join(row) + " \\\\")
    
    latex_content.extend([
        "\\bottomrule",
        "\\end{tabularx}",
        "\\end{table}"
    ])
    
    # Save table
    table_path = output_dirs['tables'] / "full_comparison.tex"
    with open(table_path, 'w') as f:
        f.write('\n'.join(latex_content))
    
    return table_path.name

def generate_performance_summary(baseline_data, enhancements_data, output_dirs):
    """Generate detailed performance summary report."""
    summary = []
    
    # Overall comparison
    summary.append("\\section*{Performance Summary}")
    summary.append("\\begin{itemize}")
    
    for enh_name, enh_data in enhancements_data.items():
        change = calculate_percentage_change(
            enh_data.get('mean_ap', 0),
            baseline_data.get('mean_ap', 0)
        )
        summary.append(
            f"\\item {ENHANCEMENT_NAMES.get(enh_name, enh_name)}: "
            f"Mean AP changed by {get_latex_color(change)} "
            f"({baseline_data.get('mean_ap', 0):.4f} → {enh_data.get('mean_ap', 0):.4f})"
        )
    
    summary.append("\\end{itemize}")
    
    # Class-specific highlights
    all_classes = set(baseline_data.get("mean_dist_aps", {}).keys())
    for enh_data in enhancements_data.values():
        all_classes.update(enh_data.get("mean_dist_aps", {}).keys())
    
    top_class = max(
        all_classes,
        key=lambda x: max(
            enh.get("mean_dist_aps", {}).get(x, 0)
            for enh in enhancements_data.values()
        )
    )
    
    summary.append("\\subsection*{Class-Specific Highlights}")
    summary.append(f"Top performing class: \\textbf{{{top_class}}}")
    summary.append("\\begin{itemize}")
    
    for enh_name, enh_data in enhancements_data.items():
        ap = enh_data.get("mean_dist_aps", {}).get(top_class, 0)
        base_ap = baseline_data.get("mean_dist_aps", {}).get(top_class, 0)
        change = calculate_percentage_change(ap, base_ap)
        summary.append(
            f"\\item {ENHANCEMENT_NAMES.get(enh_name, enh_name)}: "
            f"AP {get_latex_color(change)} "
            f"({base_ap:.4f} → {ap:.4f})"
        )
    
    summary.append("\\end{itemize}")
    
    # Error improvements
    summary.append("\\subsection*{Error Metric Improvements}")
    summary.append("Most significant error reductions:")
    summary.append("\\begin{itemize}")
    
    error_improvements = []
    for err_type in baseline_data.get("tp_errors", {}).keys():
        for enh_name, enh_data in enhancements_data.items():
            base_err = baseline_data.get("tp_errors", {}).get(err_type, float('nan'))
            enh_err = enh_data.get("tp_errors", {}).get(err_type, float('nan'))
            if not np.isnan(base_err) and not np.isnan(enh_err):
                improvement = -calculate_percentage_change(enh_err, base_err)
                error_improvements.append((
                    ENHANCEMENT_NAMES.get(enh_name, enh_name),
                    ' '.join([word.capitalize() for word in err_type.split('_')]),
                    improvement
                ))
    
    #))
    
    # Sort by improvement magnitude
    error_improvements.sort(key=lambda x: abs(x[2]), reverse=True)
    
    # Add top 3 improvements
    for enh_name, err_name, improvement in error_improvements[:3]:
        summary.append(
            f"\\item {enh_name}: {err_name} improved by {get_latex_color(improvement)}"
        )
    
    summary.append("\\end{itemize}")
    
    # Save summary
    summary_path = output_dirs['tables'] / "performance_summary.tex"
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary))
    
    return summary_path.name

def create_error_improvement_chart(baseline_data, enhancements_data, output_dirs, colors):
    """Visualize error metric improvements."""
    error_types = list(baseline_data.get("tp_errors", {}).keys())
    error_labels = [
        ' '.join([word.capitalize() for word in err.split('_')])
        for err in error_types
    ]
    
    # Calculate improvements
    improvements = {name: [] for name in enhancements_data}
    valid_errors = []
    
    for err_type, err_label in zip(error_types, error_labels):
        base_val = baseline_data.get("tp_errors", {}).get(err_type, float('nan'))
        if np.isnan(base_val):
            continue
            
        valid = False
        for enh_name, enh_data in enhancements_data.items():
            enh_val = enh_data.get("tp_errors", {}).get(err_type, float('nan'))
            if not np.isnan(enh_val):
                imp = -calculate_percentage_change(enh_val, base_val)
                improvements[enh_name].append(imp)
                valid = True
            else:
                improvements[enh_name].append(0)
        
        if valid:
            valid_errors.append(err_label)
    
    if not valid_errors:
        return None
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(valid_errors))
    width = 0.2
    
    # Plot each enhancement's improvements
    for i, (enh_name, imp_values) in enumerate(improvements.items()):
        ax.bar(
            x + i*width - width*(len(enhancements_data)-1)/2,
            imp_values[:len(valid_errors)],
            width,
            label=ENHANCEMENT_NAMES.get(enh_name, enh_name),
            color=colors[enh_name]
        )
    
    # Formatting
    ax.set_ylabel('Improvement (%)', fontweight='bold')
    ax.set_title('Error Metric Improvements vs Baseline', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(valid_errors, rotation=45, ha='right')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.legend(loc='upper right')
    ax.grid(True, axis='y', linestyle=':')
    
    # Save
    fig_path = output_dirs['figures'] / "error_improvements.png"
    plt.savefig(fig_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return fig_path.name

def create_error_heatmap(baseline_data, enhancements_data, output_dirs):
    """Create heatmap showing best enhancement for each error type."""
    error_types = list(baseline_data.get("tp_errors", {}).keys())
    classes = list(baseline_data.get("label_tp_errors", {}).keys())
    
    # Prepare data
    heatmap_data = []
    for cls in classes:
        for err_type in error_types:
            base_val = baseline_data.get("label_tp_errors", {}).get(cls, {}).get(err_type, float('nan'))
            if np.isnan(base_val):
                continue
                
            best_enh = None
            best_imp = -float('inf')
            for enh_name, enh_data in enhancements_data.items():
                enh_val = enh_data.get("label_tp_errors", {}).get(cls, {}).get(err_type, float('nan'))
                if not np.isnan(enh_val):
                    imp = -calculate_percentage_change(enh_val, base_val)
                    if imp > best_imp:
                        best_imp = imp
                        best_enh = enh_name
            
            if best_enh:
                heatmap_data.append({
                    'Class': cls,
                    'Error': ' '.join([word.capitalize() for word in err_type.split('_')]),
                    'Improvement': best_imp,
                    'Best': ENHANCEMENT_NAMES.get(best_enh, best_enh)
                })
    
    if not heatmap_data:
        return None
    
    # Create heatmap
    df = pd.DataFrame(heatmap_data)
    pivot_df = df.pivot_table(
        index='Class',
        columns='Error',
        values='Improvement',
        aggfunc='max'
    )
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        pivot_df,
        annot=True,
        fmt=".1f",
        cmap="vlag",
        center=0,
        ax=ax,
        cbar_kws={'label': 'Improvement (%)'}
    )
    
    ax.set_title('Best Error Improvements by Class and Metric', fontweight='bold')
    ax.set_xlabel('Error Type')
    ax.set_ylabel('Class')
    
    fig_path = output_dirs['figures'] / "error_heatmap.png"
    plt.savefig(fig_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return fig_path.name

def generate_all_results(baseline_path, enhancement_paths, output_dir):
    """Generate all comparison results."""
    # Load data
    baseline_data = load_json(baseline_path)
    enhancements_data = {
        f'enhancement{i+1}': load_json(path)
        for i, path in enumerate(enhancement_paths)
    }
    
    # Setup
    colors = setup_plot_style()
    output_dirs = create_output_dirs(output_dir)
    
    # Generate visualizations
    results = {}
    
    # Top classes chart
    results['top_classes'] = create_top_classes_chart(
        baseline_data, enhancements_data, output_dirs, colors
    )
    
    # Distance threshold chart for top class
    top_class = max(
        baseline_data.get("mean_dist_aps", {}).keys(),
        key=lambda x: baseline_data.get("mean_dist_aps", {}).get(x, 0)
    )
    results['distance_threshold'] = create_distance_threshold_chart(
        baseline_data, enhancements_data, top_class, output_dirs, colors
    )
    
    # Error visualizations
    results['error_improvement'] = create_error_improvement_chart(
        baseline_data, enhancements_data, output_dirs, colors
    )
    results['error_heatmap'] = create_error_heatmap(
        baseline_data, enhancements_data, output_dirs
    )
    
    # Generate reports
    results['comparison_table'] = generate_latex_comparison_table(
        baseline_data, enhancements_data, output_dirs
    )
    results['performance_summary'] = generate_performance_summary(
        baseline_data, enhancements_data, output_dirs
    )
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description='Generate comprehensive comparison of baseline with 3 enhancements'
    )
    parser.add_argument(
        '--baseline',
        required=True,
        help='Path to baseline metrics JSON file'
    )
    parser.add_argument(
        '--enhancements',
        required=True,
        nargs=3,
        help='Paths to three enhancement metrics JSON files'
    )
    parser.add_argument(
        '--output',
        default='comparison_results',
        help='Output directory path'
    )
    
    args = parser.parse_args()
    
    try:
        results = generate_all_results(
            args.baseline,
            args.enhancements,
            args.output
        )
        
        print("Successfully generated:")
        for name, path in results.items():
            print(f"- {name}: {path}")
            
    except Exception as e:
        print(f"Error generating results: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()