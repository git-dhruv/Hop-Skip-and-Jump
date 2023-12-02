import numpy as np
import matplotlib.pyplot as plt
import pickle
from os.path import join as opj

folder_location = r"/home/anirudhkailaje/Documents/01_UPenn/02_MEAM5170/03_FinalProject/src/logs/2023-12-02_17-18-44"
with open(opj(folder_location, "data.pickle"), "rb") as file:
    COM_POS, COM_VEL, T_POS, T_VEL, left, right, COM_POS_DESIRED, COM_VEL_DESIRED, Torso_POS_DESIRED, Torso_VEL_DESIRED, LFt_POS_DESIRED, RFt_POS_DESIRED, FSM, Torque, Costs, t, LContact, RContact = pickle.load(file)
FSM = np.int32(FSM)
print(FSM.shape)

fig, axs = plt.subplots(2, 1, figsize=(7, 10))
axs[0].imshow(np.row_stack((RContact,LContact))[:,1000:1500]  )
axs[1].plot(FSM[0,1000:1500]  )
plt.show()



colors = ['skyblue', 'lightcoral', 'lightgreen']
# Find boundaries where the finite state changes
boundaries = np.where(np.diff(FSM[0, :]) != 0)[0]
boundaries = np.concatenate([[0], boundaries, [len(t)-1]])

# Function to create subplots for 3D data
def plot_3d_data(ax, t, data, label, linestyle='-', alpha=0.1):
    ax.plot(t, data[0, :], label=f'{label} - X', linestyle=linestyle, color='C0')
    ax.plot(t, data[1, :], label=f'{label} - Y', linestyle=linestyle, color='C1')
    ax.plot(t, data[2, :], label=f'{label} - Z', linestyle=linestyle, color='C2')

    ## Color Coding ##
    for i in range(0, len(boundaries)-1):
        start_idx, end_idx = boundaries[i]+1, boundaries[i+1]
        ax.axvspan(t[start_idx], t[end_idx], color=colors[FSM[0, start_idx]], alpha=alpha, lw=0)

# Create subplots
fig, axs = plt.subplots(8, 1, figsize=(7, 20))

# Plotting each pair of actual and desired values
plot_3d_data(axs[0], t, COM_POS, 'COM_POS', alpha=0.1)
plot_3d_data(axs[0], t, COM_POS_DESIRED, 'COM_POS_DESIRED', '--', alpha=0.1)
axs[0].set_title('Center of Mass Position')
axs[0].legend(loc='upper right')

plot_3d_data(axs[1], t, COM_VEL, 'COM_POS', alpha=0.1)
plot_3d_data(axs[1], t, COM_VEL_DESIRED, 'COM_POS_DESIRED', '--', alpha=0.1)
axs[1].set_title('Center of Mass Velocity')
axs[1].legend(loc='upper right')

axs[2].plot(t, T_POS[0], label = 'T_POS', color='C0')
axs[2].plot(t, Torso_POS_DESIRED[0], label='Torso_POS_DESIRED', linestyle='--', color='C1')
for i in range(0, len(boundaries)-1):
    start_idx, end_idx = boundaries[i]+1, boundaries[i+1]
    axs[2].axvspan(t[start_idx], t[end_idx], color=colors[FSM[0, start_idx]], alpha=0.1, lw=0)
axs[2].set_title('Torso Position')
axs[2].legend(loc='upper right')

axs[3].plot(t, T_VEL[0], label='T_VEL', color='C0')
axs[3].plot(t, Torso_VEL_DESIRED[0], label='Torso_VEL_DESIRED', linestyle='--', color='C1')
for i in range(0, len(boundaries)-1):
    start_idx, end_idx = boundaries[i]+1, boundaries[i+1]
    axs[3].axvspan(t[start_idx], t[end_idx], color=colors[FSM[0, start_idx]], alpha=0.1, lw=0)
axs[3].set_title('Torso Velocity')
axs[3].legend(loc='upper right')

plot_3d_data(axs[4], t, left, 'Left', alpha=0.1)
plot_3d_data(axs[4], t, LFt_POS_DESIRED, 'Ft_POS_DESIRED', '--', alpha=0.1)
axs[4].set_title('Left and Desired Foot Position')
axs[4].legend(loc='upper right')

plot_3d_data(axs[5], t, right, 'Right', alpha=0.1)
plot_3d_data(axs[5], t, RFt_POS_DESIRED, 'Ft_POS_DESIRED', '--', alpha=0.1)
axs[5].set_title('Right and Desired Foot Position')
axs[5].legend(loc='upper right')


axs[6].semilogy(t, Costs[0], label='Cost', color='C0')
for i in range(0, len(boundaries)-1):
    start_idx, end_idx = boundaries[i]+1, boundaries[i+1]
    axs[6].axvspan(t[start_idx], t[end_idx], color=colors[FSM[0, start_idx]], alpha=0.1, lw=0)
axs[6].set_title('Optimal Costs')
axs[6].legend(loc='upper right')


axs[7].plot(t, Torque[0,:], label='T1', color='C0')
axs[7].plot(t, Torque[1,:], label='T2', color='C1')
axs[7].plot(t, Torque[2,:], label='T3', color='C2')
axs[7].plot(t, Torque[3,:], label='T4', color='C3')
for i in range(0, len(boundaries)-1):
    start_idx, end_idx = boundaries[i]+1, boundaries[i+1]
    axs[7].axvspan(t[start_idx], t[end_idx], color=colors[FSM[0, start_idx]], alpha=0.1, lw=0)
axs[7].set_title('Torques')
axs[7].legend(loc='upper right')


# Set common labels and title
fig.text(0.5, 0.04, 'Time', ha='center', va='center')
fig.text(0.06, 0.5, 'Position', ha='center', va='center', rotation='vertical')
fig.suptitle('Performance', fontsize=16)

plt.tight_layout()
plt.show()