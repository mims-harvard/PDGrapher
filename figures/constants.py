import seaborn as sns

max_nodes = 10716

plotting_context = {
'axes.linewidth': 1,
 'grid.linewidth': 1,
 'lines.linewidth': 1.5,
 'lines.markersize': 9.0,
 'patch.linewidth': 1,
 'xtick.major.width': 1,
 'ytick.major.width': 1,
 'xtick.minor.width': 1,
 'ytick.minor.width': 1,
 'xtick.major.size': 9.0,
 'ytick.major.size': 9.0,
 'xtick.minor.size': 6.0,
 'ytick.minor.size': 6.0,
 'font.size': 8.0,
 'axes.labelsize': 11.0,
 'axes.titlesize': 11.0,
 'xtick.labelsize': 11.0,
 'ytick.labelsize': 11.0,
 'legend.fontsize': 11,
 'legend.title_fontsize': 12}

plotting_context_small = {
'axes.linewidth': 0.5,
 'grid.linewidth': 1,
 'lines.linewidth': 1,
 'lines.markersize': 9.0,
 'patch.linewidth': 1,
 'xtick.major.width': 0.5,
 'ytick.major.width': 0.5,
 'xtick.minor.width': 0.5,
 'ytick.minor.width': 0.5,
 'xtick.major.size': 2.0,
 'ytick.major.size': 2.0,
 'xtick.minor.size': 0,
 'ytick.minor.size': 0,
 'font.size': 6.0,
 'axes.labelsize': 9.0,
 'axes.titlesize': 9.0,
 'xtick.labelsize': 9.0,
 'ytick.labelsize': 9.0,
 'legend.fontsize': 9,
 'legend.title_fontsize': 6}

color_random = '#9e9e9e'
color_chemical = '#f2bc8d'
color_genetic = '#c41795'
heatmap_palette_genetic = sns.diverging_palette(h_neg=206, h_pos=320, s=80, l=40, sep=1, n=100, center='light', as_cmap=True)
heatmap_palette_chemical = sns.diverging_palette(h_neg=206, h_pos=26, s=80, l=60, sep=1, n=100, center='light', as_cmap=True)
colors_lineplot_genetic = ['#a3598a','#c41795','#d18bbc', '#ebeaeb', '#565655', '#9e9e9e']
colors_lineplot_chemical = ['#c7a98c','#f2bc8d', '#fbe5d2', '#ebeaeb', '#565655', '#9e9e9e']