import numpy as np
import matplotlib.pyplot as plt
from celluloid import Camera

def trial_movie(neural_inputs, labels, actions=None, n_fix_tuned=4):
    # neural_inputs has shape (T, d)
    T, d = neural_inputs.shape
    T, n_outputs = labels.shape
    fig = plt.figure()
    camera = Camera(fig)

    # Input subplot
    ax1 = plt.subplot(221, projection='polar')
    ax1.set_rticks([])
    ax1.set_ylim(0,4)
    ax1.set_xticklabels([])
    ax1.set_title("Input 1")

    # Input subplot
    ax4 = plt.subplot(222, projection='polar')
    ax4.set_rticks([])
    ax4.set_ylim(0,4)
    ax4.set_xticklabels([])
    ax4.set_title("Input 2")

    # Labels subplot
    ax2 = plt.subplot(223, projection='polar')
    ax2.set_rticks([])
    ax2.set_ylim(0,1)
    ax2.set_xticklabels([])
    ax2.set_title("Labels")

    # Actions subplot
    ax3 = plt.subplot(224, projection='polar')
    ax3.set_rticks([])
    ax3.set_ylim(0,1)
    ax3.set_xticklabels([])
    ax3.set_title("Actions")

    # Prep data
    n_motion_tuned = d - n_fix_tuned
    n_motion_tuned = int(0.5*n_motion_tuned) # 2nd half is the 2nd eyeball
    n_dirs = n_outputs-1
    input_angles = [k*(2*np.pi/n_motion_tuned) for k in range(n_motion_tuned)]
    label_angles = [k*(2*np.pi/n_dirs) for k in range(n_dirs)]

    # Plot
    for t in range(T):
        # Plot input
        ax1.plot(input_angles, neural_inputs[t,:n_motion_tuned], 'b-')
        if neural_inputs[t,-1] == 4.:
            ax1.plot(0, 0, 'rx', markersize=20)
            ax1.plot(0, 0, 'r+', markersize=20)
        ax4.plot(input_angles, neural_inputs[t,n_motion_tuned:2*n_motion_tuned], 'b-')
        if neural_inputs[t,-1] == 4.:
            ax4.plot(0, 0, 'rx', markersize=20)
            ax4.plot(0, 0, 'r+', markersize=20)

        # Plot labels
        ax2.plot(label_angles, labels[t,:-1], 'g-')
        if labels[t,-1] == 1.:
            ax2.plot(0, 0, 'gx', markersize=20)
            ax2.plot(0, 0, 'g+', markersize=20)

        # Plot actions
        if actions is not None:
            ax3.plot(label_angles, actions[t,:-1], 'm-')
            if actions[t,-1] == 1.:
                ax3.plot(0, 0, 'mx', markersize=20)
                ax3.plot(0, 0, 'm+', markersize=20)

        # Capture frame for animation
        camera.snap()
    return camera.animate()
