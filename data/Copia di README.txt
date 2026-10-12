The file trajectories.csv contains the dynamics of magnetizations m_j and correlations c_ij of random initial states under the PXP model composed of N qubits.

We consider N_traj trajectories that correspond to N_traj different random initial states. For each trajectory we simulate the dynamics of a random initial state and compute at each time N magnetizations (m_1, m_2, ..., m_N) and N(N-1)/2 correlations (c_12, c_13, ..., c_1N, c_23,..., c_2N,...,c_{N-2,N-1},c_{N-2,N}.c_{N-1,N}). The dynamics is sampled at the time values 0,dt,2*dt,...,(N_points-1)*dt, where (N_points-1)*dt=T_fin is the final sampled time.

For each trajectory, the first column contains the time values (0,dt,2*dt,...,T_fin), then the magnetizations (m_1, m_2, ..., m_N) at each time and then the correlations at each time (c_12, c_13, ..., c_1N, c_23,..., c_2N,...,c_{N-2,N-1},c_{N-2,N}.c_{N-1,N}). Thus, we have total 1 + N + N(N-1)/2 = 1 + N(N+1)/2 columns. The number of rows for each trajectory is N_points.

At any 0 on the first row (times), a new trajectory starts, so the total number of rows in the file is N_points x N_traj.

We consider N=10, T_fin=20, dt = 0.02, N_points = 1001, N_traj=400; thus the number of columns is 56 and the number of rows is 400400. 
