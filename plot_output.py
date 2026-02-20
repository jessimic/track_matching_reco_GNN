from spine.io.read.hdf5 import HDF5Reader

reader = HDF5Reader('/n/netscratch/iaifi_lab/Everyone/jmicallef/tester_8epochs.h5')

data = reader[1]

from spine.utils.globals import GROUP_COL
from spine.vis.network import network_topology
from spine.vis.layout import HIGH_CONTRAST_COLORS, layout3d

from plotly import graph_objs as go
from plotly.offline import iplot

trace = []

edge_index = data['edge_index'][data['edge_pred'][:, 1] > data['edge_pred'][:, 0]]
edge_index = []
print(edge_index)

trace+= network_topology(data['data'], data['clusts'], edge_index, clust_labels=data['group_pred'], edge_labels=data['edge_pred'],
                       markersize=2, cmin=0, cmax=50, colorscale=HIGH_CONTRAST_COLORS,
                       name='')

fig = go.Figure(data=trace,layout=layout3d(meta=data['meta']))
fig.write_image("output_plot.png")
#iplot(fig)
