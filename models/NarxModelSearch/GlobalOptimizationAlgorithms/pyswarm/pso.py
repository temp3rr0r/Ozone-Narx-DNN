from functools import partial
import numpy as np
import os


def _obj_wrapper(func, args, kwargs, x):
    return func(x, *args, **kwargs)


def _is_feasible_wrapper(func, x):
    return np.all(func(x) >= 0)


def _cons_none_wrapper(x):
    return np.array([0])


def _cons_ieqcons_wrapper(ieqcons, args, kwargs, x):
    return np.array([y(x, *args, **kwargs) for y in ieqcons])


def _cons_f_ieqcons_wrapper(f_ieqcons, args, kwargs, x):
    return np.array(f_ieqcons(x, *args, **kwargs))


def pso(func, lb, ub, ieqcons=[], f_ieqcons=None, args=(), kwargs={},
        swarmsize=100, omega=0.5, phip=0.5, phig=0.5, maxiter=100,
        minstep=1e-8, minfunc=1e-8, debug=False, processes=1,
        particle_output=False, rank=0, storeCheckpoints=False, data_manipulation=None):
    """
    Perform a particle swarm optimization (PSO)

    Parameters
    ==========
    func : function
        The function to be minimized
    lb : array
        The lower bounds of the design variable(s)
    ub : array
        The upper bounds of the design variable(s)

    Optional
    ========
    ieqcons : list
        A list of functions of length n such that ieqcons[j](x,*args) >= 0.0 in
        a successfully optimized problem (Default: [])
    f_ieqcons : function
        Returns a 1-D array in which each element must be greater or equal
        to 0.0 in a successfully optimized problem. If f_ieqcons is specified,
        ieqcons is ignored (Default: None)
    args : tuple
        Additional arguments passed to objective and constraint functions
        (Default: empty tuple)
    kwargs : dict
        Additional keyword arguments passed to objective and constraint
        functions (Default: empty dict)
    swarmsize : int
        The number of particles in the swarm (Default: 100)
    omega : scalar
        Particle velocity
         scaling factor (Default: 0.5)
    phip : scalar
        Scaling factor to search away from the particle's best known position
        (Default: 0.5)
    phig : scalar
        Scaling factor to search away from the swarm's best known position
        (Default: 0.5)
    maxiter : int
        The maximum number of iterations for the swarm to search (Default: 100)
    minstep : scalar
        The minimum stepsize of swarm's best position before the search
        terminates (Default: 1e-8)
    minfunc : scalar
        The minimum change of swarm's best objective value before the search
        terminates (Default: 1e-8)
    debug : boolean
        If True, progress statements will be displayed every iteration
        (Default: False)
    processes : int
        The number of processes to use to evaluate objective function and
        constraints (default: 1)
    particle_output : boolean
        Whether to include the best per-particle position and the objective
        values at those.

    Returns
    =======
    g : array
        The swarm's best known position (optimal design)
    f : scalar
        The objective value at ``g``
    p : array
        The best known position per particle
    pf: arrray
        The objective values at each position in p

    """

    assert len(lb) == len(ub), 'Lower- and upper-bounds must be the same length'
    assert hasattr(func, '__call__'), 'Invalid function handle'
    lb = np.array(lb)
    ub = np.array(ub)
    assert np.all(ub > lb), 'All upper-bound values must be greater than lower-bound values'

    vhigh = np.abs(ub - lb)
    vlow = -vhigh

    # Initialize objective function
    obj = partial(_obj_wrapper, func, args, kwargs)

    # Check for constraint function(s) #########################################
    if f_ieqcons is None:
        if not len(ieqcons):
            if debug:
                print('No constraints given.')
            cons = _cons_none_wrapper
        else:
            if debug:
                print('Converting ieqcons to a single constraint function')
            cons = partial(_cons_ieqcons_wrapper, ieqcons, args, kwargs)
    else:
        if debug:
            print('Single constraint function given in f_ieqcons')
        cons = partial(_cons_f_ieqcons_wrapper, f_ieqcons, args, kwargs)
    is_feasible = partial(_is_feasible_wrapper, cons)

    # Initialize the multiprocessing module if necessary
    if processes > 1:
        import multiprocessing
        mp_pool = multiprocessing.Pool(processes)

    # Initialize the particle swarm ############################################
    S = swarmsize
    D = len(lb)  # the number of dimensions each particle has
    x = np.random.rand(S, D)  # particle positions
    v = np.zeros_like(x)  # particle velocities
    p = np.zeros_like(x)  # best particle positions
    fx = np.zeros(S)  # current particle function values
    fs = np.zeros(S, dtype=bool)  # feasibility of each particle
    fp = np.ones(S) * np.inf  # best particle function values
    g = []  # best swarm position
    fg = np.inf  # best swarm position starting value

    # Initialize the particle's position
    x = lb + x * (ub - lb)

    psoUuid = "rank" + str(rank)
    import pickle

    k = data_manipulation["swapEvery"]
    swap = False
    non_communicating_island = data_manipulation["non_communicating_islands"]

    # Calculate objective and constraints for each particle
    if processes > 1:
        fx = np.array(mp_pool.map(obj, x))
        fs = np.array(mp_pool.map(is_feasible, x))
    else:
        # for i in range(S):
        i = 0
        while i < S:

            # Pickle new island params
            # Read pickle
            if storeCheckpoints:
                if os.path.exists("foundModels/psoLastInitState_{}.pkl".format(psoUuid)):
                    with open("foundModels/psoLastInitState_{}.pkl".format(psoUuid), "rb") as f:
                        pickleStateInitDictionary = pickle.load(f)
                        # Now you can use the dump object as the original one
                        func = pickleStateInitDictionary["func"]
                        cons = pickleStateInitDictionary["cons"]
                        D = pickleStateInitDictionary["D"]
                        S = pickleStateInitDictionary["S"]
                        agentIn = pickleStateInitDictionary["agentIn"]
                        args = pickleStateInitDictionary["args"]
                        f_ieqcons = pickleStateInitDictionary["f_ieqcons"]
                        fg = pickleStateInitDictionary["fg"]
                        fp = pickleStateInitDictionary["fp"]
                        fs = pickleStateInitDictionary["fs"]
                        fx = pickleStateInitDictionary["fx"]
                        g = pickleStateInitDictionary["g"]
                        i = pickleStateInitDictionary["i"]
                        ieqcons = pickleStateInitDictionary["ieqcons"]
                        is_feasible = pickleStateInitDictionary["is_feasible"]
                        kwargs = pickleStateInitDictionary["kwargs"]
                        lb = pickleStateInitDictionary["lb"]
                        maxiter = pickleStateInitDictionary["maxiter"]
                        minfunc = pickleStateInitDictionary["minfunc"]
                        minstep = pickleStateInitDictionary["minstep"]
                        obj = pickleStateInitDictionary["obj"]
                        omega = pickleStateInitDictionary["omega"]
                        p = pickleStateInitDictionary["p"]
                        particle_output = pickleStateInitDictionary["particle_output"]
                        phig = pickleStateInitDictionary["phig"]
                        phip = pickleStateInitDictionary["phip"]
                        processes = pickleStateInitDictionary["processes"]
                        swarmsize = pickleStateInitDictionary["swarmsize"]
                        ub = pickleStateInitDictionary["ub"]
                        v = pickleStateInitDictionary["v"]
                        vhigh = pickleStateInitDictionary["vhigh"]
                        vlow = pickleStateInitDictionary["vlow"]
                        x = pickleStateInitDictionary["x"]
                        print(
                            "-- Checkpoint Init recovered (rank: {}, i: {}): last x: {}".format(rank, i, x[i - 1, :]))

            if i < S:
                fx[i], data_worker_to_master = obj(x[i, :])

                fs[i] = is_feasible(x[i, :])

                # Always send the best agent back
                # Worker to master
                i_min = np.argmin(fx)
                data_worker_to_master["mean_mse"] = fx[i_min]
                data_worker_to_master["agent"] = x[i_min, :]
                comm = data_manipulation["comm"]
                req = comm.isend(data_worker_to_master, dest=0, tag=1)  # Send data async to master
                req.wait()
                # Master to worker
                data_master_to_worker = comm.recv(source=0, tag=2)  # Receive data sync (blocking) from master
                # Replace worst agent
                if i % k == 0 and i > 0 and not non_communicating_island:  # Send back found agent
                    swap = True
                if swap and data_master_to_worker["iteration"] >= (int(i / k) * k):
                    print(
                        "========= Swapping (ranks: from-{}-to-{})... (iteration: {}, every: {}, otherIteration: {})".format(
                            data_master_to_worker["fromRank"], data_worker_to_master["rank"], i, k,
                            data_master_to_worker["iteration"]))
                    received_agent = data_master_to_worker["agent"]
                    i_max = np.argmax(fx)
                    x[i_max, :] = received_agent
                    fx[i_min] = data_master_to_worker["mean_mse"]
                    swap = False

                i += 1

                # Save pickle
                if storeCheckpoints:
                    with open("foundModels/psoLastInitState_{}.pkl".format(psoUuid), "wb") as f:
                        pickleStateInitDictionary = {"func": func, "cons": cons, "D": D, "S": S, "agentIn": agentIn,
                                                     "args": args,
                                                     "f_ieqcons": f_ieqcons, "fg": fg, "fp": fp, "fs": fs, "fx": fx,
                                                     "g": g, "i": i,
                                                     "ieqcons": ieqcons, "is_feasible": is_feasible, "kwargs": kwargs,
                                                     "lb": lb,
                                                     "maxiter": maxiter, "minfunc": minfunc, "minstep": minstep,
                                                     "obj": obj,
                                                     "omega": omega, "p": p, "particle_output": particle_output,
                                                     "phig": phig,
                                                     "phip": phip, "processes": processes, "swarmsize": swarmsize,
                                                     "ub": ub, "v": v,
                                                     "vhigh": vhigh, "vlow": vlow, "x": x}
                        pickle.dump(pickleStateInitDictionary, f, pickle.HIGHEST_PROTOCOL)
                        print(
                            "-- Checkpoint Init stored (rank: {}, i: {}): last x: {}".format(rank, i, x[i - 1, :]))

    # Store particle's best position (if constraints are satisfied)
    i_update = np.logical_and((fx < fp), fs)
    p[i_update, :] = x[i_update, :].copy()
    fp[i_update] = fx[i_update]

    # Update swarm's best position
    i_min = np.argmin(fp)
    if fp[i_min] < fg:
        fg = fp[i_min]
        g = p[i_min, :].copy()
    else:
        # At the start, there may not be any feasible starting point, so just
        # give it a temporary "best" point since it's likely to change
        g = x[0, :].copy()

    # Initialize the particle's velocity
    v = vlow + np.random.rand(S, D) * (vhigh - vlow)

    # Iterate until termination criterion met ##################################
    it = 1
    while it <= maxiter:
        rp = np.random.uniform(size=(S, D))
        rg = np.random.uniform(size=(S, D))

        # Update the particles velocities
        v = omega * v + phip * rp * (p - x) + phig * rg * (g - x)
        # Update the particles' positions
        x = x + v
        # Correct for bound violations
        maskl = x < lb
        masku = x > ub
        x = x * (~np.logical_or(maskl, masku)) + lb * maskl + ub * masku

        # Update objectives and constraints
        if processes > 1:
            fx = np.array(mp_pool.map(obj, x))
            fs = np.array(mp_pool.map(is_feasible, x))
        else:
            # for i in range(S):
            i = 0
            while i < S:

                # Read pickle
                if storeCheckpoints:
                    if os.path.exists("foundModels/psoLastIterationState_{}.pkl".format(psoUuid)):
                        with open("foundModels/psoLastIterationState_{}.pkl".format(psoUuid), "rb") as f:
                            pickleStateIterationsDictionary = pickle.load(f)
                            # Now you can use the dump object as the original one
                            func = pickleStateIterationsDictionary["func"]
                            cons = pickleStateIterationsDictionary["cons"]
                            D = pickleStateIterationsDictionary["D"]
                            S = pickleStateIterationsDictionary["S"]
                            agentIn = pickleStateIterationsDictionary["agentIn"]
                            args = pickleStateIterationsDictionary["args"]
                            f_ieqcons = pickleStateIterationsDictionary["f_ieqcons"]
                            fg = pickleStateIterationsDictionary["fg"]
                            fp = pickleStateIterationsDictionary["fp"]
                            fs = pickleStateIterationsDictionary["fs"]
                            fx = pickleStateIterationsDictionary["fx"]
                            g = pickleStateIterationsDictionary["g"]
                            i = pickleStateIterationsDictionary["i"]
                            i_min = pickleStateIterationsDictionary["i_min"]
                            i_update = pickleStateIterationsDictionary["i_update"]
                            ieqcons = pickleStateIterationsDictionary["ieqcons"]
                            is_feasible = pickleStateIterationsDictionary["is_feasible"]
                            it = pickleStateIterationsDictionary["it"]
                            kwargs = pickleStateIterationsDictionary["kwargs"]
                            lb = pickleStateIterationsDictionary["lb"]
                            maxiter = pickleStateIterationsDictionary["maxiter"]
                            minfunc = pickleStateIterationsDictionary["minfunc"]
                            minstep = pickleStateIterationsDictionary["minstep"]
                            obj = pickleStateIterationsDictionary["obj"]
                            omega = pickleStateIterationsDictionary["omega"]
                            p = pickleStateIterationsDictionary["p"]
                            p = pickleStateIterationsDictionary["p"]
                            particle_output = pickleStateIterationsDictionary["particle_output"]
                            phig = pickleStateIterationsDictionary["phig"]
                            rg = pickleStateIterationsDictionary["rg"]
                            rp = pickleStateIterationsDictionary["rp"]
                            phip = pickleStateIterationsDictionary["phip"]
                            processes = pickleStateIterationsDictionary["processes"]
                            swarmsize = pickleStateIterationsDictionary["swarmsize"]
                            ub = pickleStateIterationsDictionary["ub"]
                            v = pickleStateIterationsDictionary["v"]
                            vhigh = pickleStateIterationsDictionary["vhigh"]
                            vlow = pickleStateIterationsDictionary["vlow"]
                            x = pickleStateIterationsDictionary["x"]
                            print("-- Checkpoint Iteration (it: {}) recovered (rank: {}, i: {}): last x: {}".format(
                                it, rank, i, x[i - 1, :]))

                if i < S:
                    fx[i], data_worker_to_master = obj(x[i, :])

                    fs[i] = is_feasible(x[i, :])

                    # Always send the best agent back
                    # Worker to master
                    i_min = np.argmin(fx)
                    data_worker_to_master["mean_mse"] = fx[i_min]
                    data_worker_to_master["agent"] = x[i_min, :]
                    comm = data_manipulation["comm"]
                    req = comm.isend(data_worker_to_master, dest=0, tag=1)  # Send data async to master
                    req.wait()
                    # Master to worker
                    data_master_to_worker = comm.recv(source=0, tag=2)  # Receive data sync (blocking) from master
                    # Replace worst agent
                    if i % k == 0 and i > 0 and not non_communicating_island:  # Send back found agent
                        swap = True
                    if swap and data_master_to_worker["iteration"] >= (int(i / k) * k):
                        print(
                            "========= Swapping (ranks: from-{}-to-{})... (iteration: {}, every: {}, otherIteration: {})".format(
                                data_master_to_worker["fromRank"], data_worker_to_master["rank"], i, k,
                                data_master_to_worker["iteration"]))
                        received_agent = data_master_to_worker["agent"]
                        i_max = np.argmax(fx)
                        x[i_max, :] = received_agent
                        fx[i_min] = data_master_to_worker["mean_mse"]
                        swap = False

                    i += 1

                    # Save pickle
                    if storeCheckpoints:
                        with open("foundModels/psoLastIterationState_{}.pkl".format(psoUuid), "wb") as f:
                            pickleStateIterationsDictionary = {"func": func, "cons": cons, "D": D, "S": S,
                                                               "agentIn": agentIn, "args": args,
                                                               "f_ieqcons": f_ieqcons, "fg": fg, "fp": fp, "fs": fs,
                                                               "fx": fx, "g": g, "i": i,
                                                               "i_min": i_min, "i_update": i_update, "it": it,
                                                               "ieqcons": ieqcons, "is_feasible": is_feasible,
                                                               "kwargs": kwargs,
                                                               "maskl": maskl, "masku": masku,
                                                               "lb": lb,
                                                               "maxiter": maxiter, "minfunc": minfunc,
                                                               "minstep": minstep,
                                                               "obj": obj,
                                                               "omega": omega, "p": p,
                                                               "particle_output": particle_output,
                                                               "phig": phig,
                                                               "phip": phip, "rg": rg, "rp": rp,
                                                               "processes": processes, "swarmsize": swarmsize, "ub": ub,
                                                               "v": v,
                                                               "vhigh": vhigh, "vlow": vlow, "x": x}
                            pickle.dump(pickleStateIterationsDictionary, f, pickle.HIGHEST_PROTOCOL)
                            print(
                                "-- Checkpoint Iteration (it: {}) stored (rank: {}, i: {}): last x: {}".format(
                                    it, rank, i, x[i - 1, :]))

        # Store particle's best position (if constraints are satisfied)
        i_update = np.logical_and((fx < fp), fs)
        p[i_update, :] = x[i_update, :].copy()
        fp[i_update] = fx[i_update]

        # Compare swarm's best position with global best position
        i_min = np.argmin(fp)
        if fp[i_min] < fg:
            if debug:
                print('New best for swarm at iteration {:}: {:} {:}' \
                      .format(it, p[i_min, :], fp[i_min]))

            p_min = p[i_min, :].copy()
            stepsize = np.sqrt(np.sum((g - p_min) ** 2))

            if np.abs(fg - fp[i_min]) <= minfunc:
                print('Stopping search: Swarm best objective change less than {:}' \
                      .format(minfunc))
                if particle_output:
                    return p_min, fp[i_min], p, fp
                else:
                    return p_min, fp[i_min]
            elif stepsize <= minstep:
                print('Stopping search: Swarm best position change less than {:}' \
                      .format(minstep))
                if particle_output:
                    return p_min, fp[i_min], p, fp
                else:
                    return p_min, fp[i_min]
            else:
                g = p_min.copy()
                fg = fp[i_min]

        if debug:
            print('Best after iteration {:}: {:} {:}'.format(it, g, fg))

        it += 1
        i = 0

        # Save pickle

        if storeCheckpoints:
            with open("foundModels/psoLastIterationState_{}.pkl".format(psoUuid), "wb") as f:
                pickleStateIterationsDictionary = {"func": func, "cons": cons, "D": D, "S": S, "agentIn": agentIn,
                                                   "args": args,
                                                   "f_ieqcons": f_ieqcons, "fg": fg, "fp": fp, "fs": fs, "fx": fx,
                                                   "g": g,
                                                   "i": i,
                                                   "i_min": i_min, "i_update": i_update, "it": it,
                                                   "ieqcons": ieqcons, "is_feasible": is_feasible, "kwargs": kwargs,
                                                   "maskl": maskl, "masku": masku,
                                                   "lb": lb,
                                                   "maxiter": maxiter, "minfunc": minfunc, "minstep": minstep,
                                                   "obj": obj,
                                                   "omega": omega, "p": p, "particle_output": particle_output,
                                                   "phig": phig,
                                                   "phip": phip, "rg": rg, "rp": rp,
                                                   "processes": processes, "swarmsize": swarmsize, "ub": ub, "v": v,
                                                   "vhigh": vhigh, "vlow": vlow, "x": x}
                pickle.dump(pickleStateIterationsDictionary, f, pickle.HIGHEST_PROTOCOL)
                print("-- Checkpoint Iteration End (it: {}) stored (rank: {}, i: {}): last x: {}".format(
                    it, rank, i, x[i - 1, :]))

    print('Stopping search: maximum iterations reached --> {:}'.format(maxiter))

    if not is_feasible(g):
        print("However, the optimization couldn't find a feasible design. Sorry")
    if particle_output:
        return g, fg, p, fp
    else:
        return g, fg
