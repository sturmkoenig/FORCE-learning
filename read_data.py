import numpy as np
import pylab as pl
import h5py


# select data fields:

file_base = 'data/evoked_response/20140505/evoked_response/MUA/control/ch'

data_field_name = 'mean_MUA_data'

# mapping from channels to layers
#                     LS2     LS1     PS      RS1     RS2
selected_channels = [['18',   '34',   '50',   '66',   '82'],  # L2/3
                     ['20',   '36',   '52',   '68',   '84'],  # L4
                     ['23',   '39',   '55',   '71',   '87'],  # L5
                     ['26',   '42',   '58',   '74',   '90']]  # L6


def read_data():

    num_columns = 5
    num_layers = 4
    num_time_steps = 300

    data = [[None]*num_layers]*num_columns
    # data = np.zeros([num_columns, num_layers, num_time_steps])

    for column in range(num_columns):
        for layer in range(num_layers):

            ch = selected_channels[layer][column]

            file_name = file_base+ch+'.mat'

            f = h5py.File(file_name, 'r')
            arrays = { k:v for k,v in f.items() }
            data[column][layer] = arrays[data_field_name]

    return data


def get_data_distribution(time_grid, response, dt=0.010, dv=100., vmin=-1000., vmax=1000.):

    N = int((time_grid[-1] - time_grid[0])/dt)
    M = int((vmax - vmin)/dv)-1

    dist = np.zeros([N,M])

    for i in range(N):
        t_start = np.where(time_grid >= i*dt)[0][0]
        t_end = np.where(time_grid >= (i+1)*dt)[0][0]
        dist[i,:] = np.histogram( response[t_start:t_end,:].reshape(-1), np.arange(vmin,vmax,dv) )[0]

    return dist


if __name__=='__main__':

    num_columns = 5
    num_layers = 4

    data = read_data()

    for column in range(num_columns):
        for layer in range(num_layers):
            pl.subplot(num_columns,num_layers,column*num_layers+layer+1)
            pl.plot(data[column,layer,:])

    pl.show()

