# CV_autoencoding_ligand_binding
Collective Variable Discovery by Deep Encoder-Decoder Model

Here, in this repository, time-evolution of three different trajectories have been shown
with visualization of the snap-shots in vmd along with time-evolution on 
free energy surface. 

a_traj1_4H_6H_vmd_fes.mp4 is for Trajectory-1(a) ---> Through Helix-4 and Helix-6 binding.

The lig-binding-ae-model.py python file will take the ligand-com-to-C-alpha distances 
as input vector and after model training and prediction, it will return us the 
hidden-projected "encoded data". This "encoded data" will be needed for all our
future work.

We need to have pre-installed keras and tensorflow library to run our code.
