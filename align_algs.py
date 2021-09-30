import numpy as np
import numba
import time
import librosa as lb
import multiprocessing
import os.path
import subprocess
import pickle
import logging
import sys
import pandas as pd


def align_batch(system, querylist, featdir1, featdir2, outdir, n_cores=8, downsample=1, verbose=False, **kwargs):
    '''
    Wrapper function for batch alignment that accepts a parameter for the baseline system number instead of parameters for steps and weights
    '''
    
    if system==1:
        # Baseline 1: DTW with transitions (1,1), (1,2), (2,1) and weights 2, 3, 3
        steps = np.array([1,1,1,2,2,1]).reshape((-1,2))
        weights = np.array([2,3,3])
        inputs = alignDTW_batch(querylist, featdir1, featdir2, outdir, n_cores, steps, weights, downsample, verbose)
        
    elif system==2:
        # Baseline 2: DTW with transitions (1,1), (1,2), (2,1) and weights 1, 1, 1
        steps = np.array([1,1,1,2,2,1]).reshape((-1,2))
        weights = np.array([1,1,1])
        inputs = alignDTW_batch(querylist, featdir1, featdir2, outdir, n_cores, steps, weights, downsample, verbose)
        
    elif system==3:
        # Baseline 6: NWTW
        gamma = 0.8
        inputs = alignNW_batch(querylist, featdir1, featdir2, outdir, n_cores, downsample, gamma)
        
    else:
        logging.error('Unrecognized baseline ID %s' % system)
        sys.exit(1)
        
    return



def alignDTW_batch(querylist, featdir1, featdir2, outdir, n_cores, steps, weights, downsample, verbose = False, subseq = False, librosa = True):
    start = time.time()
    outdir.mkdir(parents=True, exist_ok=True)
    
    # prep inputs for parallelization
    inputs = []
    with open(querylist, 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            assert len(parts) == 2
            featfile1 = (featdir1 / parts[0]).with_suffix('.npy')
            featfile2 = (featdir2 / parts[1]).with_suffix('.npy')
            queryid = os.path.basename(parts[0]) + '__' + os.path.basename(parts[1])
            outfile = (outdir / queryid).with_suffix('.pkl')
            
            if os.path.exists(outfile):
                print(f"Skipping {outfile}")
            else:
                inputs.append((featfile1, featfile2, steps, weights, downsample, subseq, outfile, verbose, librosa))

    # process files in parallel
    with multiprocessing.get_context("spawn").Pool(processes = n_cores) as pool:
        pool.starmap(alignDTW, inputs)
    print("Time to finish alignDTW: ", time.time() - start)
    return



@numba.jit(forceobj=True)
def alignDTW(featfile1, featfile2, steps, weights, downsample, subseq = False, outfile = None, verbose = False, librosa = True):
    '''
    Aligns featfile1 and featfile2
    
    Arguments:
    subsequence -- if True, runs subsequenceDTW instead of DTW
    verbose -- if True, prints statements for each file while processing
    librosa -- if True, uses librosa's implementation of DTW
    '''
    if verbose:
        print(outfile)
    
    # Read in chroma features from feature files
    F1 = np.load(featfile1) # 12 x N
    F2 = np.load(featfile2) # 12 x M
    
    if verbose:
        print('Read chroma features')
        print(F1)
        print(F2)

    # Make sure there is a valid path possible. If one file is over twice as long as the other one, 
    # then no valid path is possible (assuming our steps only let us move a max of 2 spaces)
    if max(F1.shape[1], F2.shape[1]) / min(F1.shape[1], F2.shape[1]) >= 2: # no valid path possible
        if verbose:
            print('Not valid')
        if outfile:
            pickle.dump(None, open(outfile, 'wb'))
        return None
    
    if verbose:
        print('Checked if valid')
    
    # For some reason, this calculation stalls for some pairs. Need to investigate more
    C = 1 - F1[:,0::downsample].T @ F2[:,0::downsample] # cos distance metric
    
    if verbose:
        print('Calculated cost matrix if enabled')
    
    # Run DTW algorithm
    
    # This is Librosa's implementation
    if librosa:
        D, wp = lb.sequence.dtw(C=C, step_sizes_sigma = steps, weights_add = weights, subseq = subseq)
#         D, wp = lb.sequence.dtw(X=F1, Y=F2, metric='cosine', step_sizes_sigma = steps, weights_add = weights, subseq = subseq)
        wp = wp[::-1] # Need to reverse path for lb
    
        # if N > M, Y can be a subsequence of X, librosa switches C to C transpose, so we want to switch the rows and columns of wp back
        if subseq and (F1.shape[1] > F2.shape[1]):
            wp = np.fliplr(wp)
            
    else:
    # This is our implementation. Currently needs cost matrix as input to run
        optcost, wp = DTW(C, steps, weights, subseq = subseq)
    if verbose:
        print('Ran DTW')
    
    # If output file is specified, save results
    if outfile:
        costs = []
        for element in wp:
            costs.append(C[element[0], element[1]])
        d = {"wp": wp, "costs": costs}
        pickle.dump(d, open(outfile, 'wb'))
        if verbose:
            print('Pickle dump')
    else:
        return wp



@numba.jit(nopython=True, cache=True)
def DTW(C, steps, weights, subseq = False):
    '''
    Find the optimal subsequence path through cost matrix C.
    
    Arguments:
    C -- cost matrix of dimension (# query frames, # reference frames)
    steps -- a numpy matrix specifying the allowable transitions.  It should be of
            dimension (L, 2), where each row specifies (row step, col step)
    weights -- a vector of size L specifying the multiplicative weights associated 
                with each of the allowable transitions
    subsequence -- if True, runs subsequence DTW instead of regular DTW
                
    Returns:
    optcost -- the optimal subsequence path score
    path -- a matrix with 2 columns specifying the optimal subsequence path.  Each row 
            specifies the (row, col) coordinate.
    '''
    D = np.zeros(C.shape)
    B = np.zeros(C.shape, dtype=np.int8)
    if subseq:
        D[0,:] = C[0,:]
    else:
        D[0,0] = C[0,0]
    for row in range(1,C.shape[0]):
        for col in range(C.shape[1]):
            mincost = np.inf
            minidx = -1
            for stepidx, step in enumerate(steps):
                (rstep, cstep) = step
                prevrow = row - rstep
                prevcol = col - cstep
                if prevrow >= 0 and prevcol >= 0:
                    pathcost = D[prevrow, prevcol] + C[row, col] * weights[stepidx]
                    if pathcost < mincost:
                        mincost = pathcost
                        minidx = stepidx
            D[row, col] = mincost
            B[row, col] = minidx
            
    if subseq:
        optcost = np.min(D[-1,:])
    else:
        optcost = D[-1,-1]
    
    path = backtrace(D, B, steps)
    path.reverse()
    path = np.array(path)
    
    return optcost, path


@numba.jit(nopython=True, cache=True)
def backtrace(D, B, steps):
    '''
    Backtraces through the cumulative cost matrix D.
    
    Arguments:
    D -- cumulative cost matrix
    B -- backtrace matrix
    steps -- a numpy matrix specifying the allowable transitions.  It should be of
            dimension (L, 2), where each row specifies (row step, col step)
    
    Returns:
    path -- a python list of (row, col) coordinates for the optimal path.
    '''
    # initialization
    r = B.shape[0] - 1
    c = B.shape[1] - 1
    path = [[r, c]]
    
    # backtrace
    while r > 0:
        step = steps[B[r, c]]
        r = r - step[0]
        c = c - step[1]
        if r != B.shape[0] - 1:
            path.append([r, c])
            
    return path


@numba.jit(nopython=True, cache=True)
def NWTW(C, gamma):
    '''
    Needleman-Wunsch time warping algorithm
    
    Arguments:
    C -- cost matrix
    gamma -- gap penalty
    '''
    NW = np.zeros(C.shape)
    B = np.zeros(C.shape, dtype=np.int8) # backtrace matrix
    
    # initialization
    for j in range(1, C.shape[1]):
        NW[0, j] = gamma + NW[0, j-1]
        B[0, j] = 5
    for i in range(1, C.shape[0]):
        NW[i, 0] = gamma + NW[i-1, 0]
        B[i, 0] = 4
        
    # dynamic programming
    for i in range(1, C.shape[0]):
        for j in range(1, C.shape[1]):
            cost_1 = C[i, j] + NW[i-1, j-1]
            cost_2 = C[i, j] + C[i, j-1] + NW[i-1, j-2]
            cost_3 = C[i, j] + C[i-1, j] + NW[i-2, j-1]
            cost_4 = gamma + NW[i-1, j]
            cost_5 = gamma + NW[i, j-1]
            costs = np.array([cost_1, cost_2, cost_3, cost_4, cost_5])
            NW[i, j] = np.min(costs)
            B[i, j] = np.argmin(costs) + 1
            
    # return cost and path (from backtrace function)
    optcost = NW[-1, -1]
    path, costs = backtrace_nwtw(NW, B, C, gamma)
    return optcost, path, costs



@numba.jit(nopython=True, cache=True)
def backtrace_nwtw(NW, B, C, gamma):
    '''
    backtrace function for NWTW
    '''
    # initialization
    r = B.shape[0] - 1
    c = B.shape[1] - 1
    path = [[r, c]]
    costs = [C[r,c]]
    steps = {1: (1, 1), 2: (1, 2), 3: (2, 1), 4: (1, 0), 5: (0, 1)}
    
    # backtrace
    while r > 0:
        step = steps[B[r, c]]
        r = r - step[0]
        c = c - step[1]
        if r != B.shape[0] - 1:
            path.append([r, c])
            
            if B[r,c] == 4 or B[r,c] == 5:
                costs.append(gamma)
            else:
                costs.append(C[r, c])
            
    return path, costs



def alignNW_batch(querylist, featdir1, featdir2, outdir, n_cores, downsample, gamma):
    start = time.time()
    outdir.mkdir(parents=True, exist_ok=True)
    
    # prep inputs for parallelization
    inputs = []
    with open(querylist, 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            assert len(parts) == 2
            featfile1 = (featdir1 / parts[0]).with_suffix('.npy')
            featfile2 = (featdir2 / parts[1]).with_suffix('.npy')
            queryid = os.path.basename(parts[0]) + '__' + os.path.basename(parts[1])
            outfile = (outdir / queryid).with_suffix('.pkl')
            
            if os.path.exists(outfile):
                print(f"Skipping {outfile}")
            else:
                inputs.append((featfile1, featfile2, downsample, gamma, outfile))
    
    # process files in parallel
    with multiprocessing.get_context("spawn").Pool(processes = n_cores) as pool:
        pool.starmap(alignNW, inputs)
    print("Time to finish align with NWTW: ", time.time() - start)
    return



@numba.jit(forceobj=True)
def alignNW(featfile1, featfile2, downsample, gamma, outfile=None):
    '''Aligns featfile1 and featfile2 using NWTW'''
    # Read in chroma features from feature files
    F1 = np.load(featfile1)  # 12 x N
    F2 = np.load(featfile2)  # 12 x M

#     # Make sure there is a valid path possible. If one file is over twice as long as the other one, 
#     # then no valid path is possible (assuming our steps only let us move a max of 2 spaces)
#     if max(F1.shape[1], F2.shape[1]) / min(F1.shape[1], F2.shape[1]) >= 2: # no valid path possible
#         if outfile:
#             pickle.dump(None, open(outfile, 'wb'))
#         return None
    
    C = 1 - F1[:,0::downsample].T @ F2[:,0::downsample] # cos distance metric
    
    # Run NWTW algorithm
    optcost, wp, costs = NWTW(C, gamma)
    wp.reverse()
    costs.reverse()
    
    # If output file is specified, save results
    if outfile:
#         costs = []
#         for element in wp:
#             costs.append(C[element[0], element[1]])
        d = {"wp": np.array(wp), "costs": np.array(costs)}
        pickle.dump(d, open(outfile, 'wb'))
    
    return wp


def align_HPTW_batch(querylist, featdir1, featdir2, outdir, n_cores, steps, weights, gamma, beta, downsample):
    
    start = time.time()
    outdir.mkdir(parents=True, exist_ok=True)
    
    # prep inputs for parallelization
    inputs = []
    with open(querylist, 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            assert len(parts) == 2
            featfile1 = (featdir1 / parts[0]).with_suffix('.npy')
            featfile2 = (featdir2 / parts[1]).with_suffix('.npy')
            queryid = os.path.basename(parts[0]) + '__' + os.path.basename(parts[1])
            outfile = (outdir / queryid).with_suffix('.pkl')
            
            if os.path.exists(outfile):
                print(f"Skipping {outfile}")
            else:
                #(featfile1, featfile2, steps, weights, gamma, beta, downsample, outfile = None, verbose = False)
                inputs.append((featfile1, featfile2, steps, weights, gamma, beta, downsample, outfile))

    # process files in parallel
    # This line of code is very important! If you do not explicitly specify the context
    # multiprocessing may stall due to how the library copies over threads from processes
    # (the processes inherit locks but not threads! So a lock might be inherited and never unlocked)
    # This forces us to create a new process from scratch so we don't have to worry about the lock problem
    with multiprocessing.get_context("spawn").Pool(processes = n_cores) as pool:
        pool.starmap(align_HPTW, inputs)
    print("Time to finish align_HPTW_batch: ", time.time() - start)
    return

@numba.jit(forceobj=True)
def align_HPTW(featfile1, featfile2, steps, weights, c_gamma, c_beta, downsample, outfile = None):
    
    # Read in features from feature files
    F1 = np.load(featfile1)
    F2 = np.load(featfile2)
        
    #C = 1 - F1[:,0::downsample].T @ F2[:,0::downsample] # cos distance metric
    C = 1 - F1[:,0:].T @ F2[:,0:] # cos distance metric
        
    # Calculate the minimum of each row in pairwise cost matrix C, and calculate the minimum of each column in C.  
    # Concatenate the two lists and determine the median of the concatenated list.  
    # Then multiply the median by c_gamma.    
    min_Crow = np.min(C, axis=1)
    min_Ccol = np.min(C, axis=0)
    min_array = np.concatenate((min_Crow, min_Ccol))
    median_min = np.median(min_array)
    gamma = median_min * c_gamma + 1e-9 # this is in case we want to align to ourselves
    beta = c_beta * gamma
        
    # Run HSTW
    D, B = HPTW(C, steps, weights, gamma, beta)
    path_v, path_h, total_path, costs = backtrace_HPTW(D, B, steps, C, gamma)
        
    # If output file is specified, save results
    
    if outfile:
#         for element in total_path:
#             if element[2] == 0:
#                 costs.append(gamma)
#             elif element[2] == 1:
#                 costs.append(C[element[0], element[1]])
                
        d = {"wp": total_path, "path_v": path_v, "path_h": path_h, "costs": costs}
        pickle.dump(d, open(outfile, 'wb'))

    return path_v, path_h

@numba.jit(nopython=True, cache=True)
def HPTW(C, steps, weights, gamma, beta):
    D = np.zeros((2, C.shape[0], C.shape[1]))
    B = np.zeros((2, C.shape[0], C.shape[1]), dtype = np.int8)
    
    # 0 = hidden, 1 = visible
    
    # Initialize
    D[0, 0, 0] = 0
    D[1, 0, 0] = np.inf
    
    for row in range(C.shape[0]):
        for col in range(C.shape[1]):
            
            # Initialize
            if row == 0 and col == 0:
                D[0, 0, 0] = gamma
                D[1, 0, 0] = C[0,0]
                continue
                
            # Hidden (assuming we only have three defined steps)
            # 0: D(0,i-1,j) + gamma (vertical)
            # 1: D(0,i,j-1) + gamma (horizontal)
            # 2: D(1,i-1,j-1) + 2*gamma (switch planes, visible to hidden)
                                    
            # We also need to check for out of bounds!
            if row < 1:
                D[0, row, col] = D[0, row, col-1] + gamma
                B[0, row, col] = 1
            elif col < 1:
                D[0, row, col] = D[0, row-1, col] + gamma
                B[0, row, col] = 0
            else:
                costs_h = np.array([D[0, row-1, col] + gamma, D[0, row, col-1] + gamma, D[1, row-1, col-1] + 2*gamma])
                D[0, row, col] = np.min(costs_h)
                B[0, row, col] = np.argmin(costs_h)
            
            
            # Visible (variable steps and weights)
            # Check which step has the minimum cost associated with it
            mincost = np.inf
            minidx = -1
            for stepidx, step in enumerate(steps):
                (rstep, cstep) = step
                prevrow = row - rstep
                prevcol = col - cstep
                
                if prevrow >= 0 and prevcol >= 0:
                    if D[1, prevrow, prevcol] + C[row, col] * weights[stepidx] < mincost:
                        mincost = D[1, prevrow, prevcol] + C[row, col] * weights[stepidx]
                        minidx = stepidx
                        
            # Also check option to come up from hidden plane to visible plane
            # Cost from Hidden to Visible: D(0,i,j) + beta 
            if D[0, row, col] + beta  < mincost:
                mincost = D[0, row, col] + beta 
                minidx = len(steps) # Reserve len(steps) as switch from hidden to visible plane
                
            D[1, row, col] = mincost
            B[1, row, col] = minidx
            
    
    return D, B

#@numba.jit(nopython=True, cache=True)
def backtrace_HPTW(D, B, steps, C, gamma):
    # Hidden = 0, Visible = 1
    
    row = D.shape[1] - 1
    col = D.shape[2] - 1
    
    # Start in plane which has lowest cost
    starting_cost = np.array([D[0, row, col], D[1, row, col]])
    plane = np.argmin(starting_cost)
    
    path_v = []
    path_h = []
    total_path = []
    costs = []
#     planes_v = []
#     planes_h = []
#     i = 0
    
    while row > 0 and col > 0:
        
        # Record row and column in path
        if plane == 0:
            path_h.append([row, col])
#             planes_h.append(i)
        else:
            path_v.append([row, col])
        costs.append(C[row, col])
#             planes_v.append(i)
        total_path.append([row, col, plane])
#         i += 1
            
        # Backtrace
        
        # Hidden plane (assuming we only have three defined steps)
        # 0: D(0,i-1,j) + gamma (vertical)
        # 1: D(0,i,j-1) + gamma (horizontal)
        # 2: D(1,i-1,j-1) + 2*gamma (switch planes, visible to hidden)
        if plane == 0:
            if B[plane, row, col] == 0: # Move back one row
                row -= 1
            elif B[plane, row, col] == 1: # Move back one column
                col -= 1
            else: # Move to visible plane and back one row and col
                plane = 1
                row -= 1
                col -= 1
                
        # Visible plane 
        # idx corrresponds to steps specified in steps array
        # last idx corresponds to switching planes
        else:
            stepidx = B[plane, row, col]
            
            if stepidx == len(steps): # Switch planes
                plane = 0
            else:
                #print(steps[stepidx])
                (rstep, cstep) = steps[stepidx]
                row = row - rstep
                col = col - cstep
                
    if plane == 0:
        path_h.append([row, col])
#         planes_h.append(i)
    else:
        path_v.append([row, col])
    costs.append(C[row, col])
#         planes_v.append(i)
    total_path.append([row, col, plane])
                
    return np.flipud(path_v), np.flipud(path_h), np.flipud(total_path), np.flipud(costs)
