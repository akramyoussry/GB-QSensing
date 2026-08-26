import csv,pickle
import numpy as np
from QubitMLModel import qubitMLmodel
import matplotlib.pyplot as plt
###############################################################################
R = 5e6 # numbre of repeations
###############################################################################
def create_model(architecture, model_name, maximums):
    """
    This function loads a pre-trained ML mode
    
    architecture: one of "WB", "GB", "BB_small" or ""BB_large"" 
    model_name: filename to load the tranied weights
    maximums: a list of all the maximums of the dataset for normalization

    """
    tau_max, fb_max, phi_max, T_max, P0_max, Pcl_max = maximums
    
    if architecture=="WB":
        mlmodel = qubitMLmodel(fb_max, tau_max, Pcl_max, P0_max, architecture="WB")
    
    elif architecture=="BB_small":
        mlmodel = qubitMLmodel(fb_max, tau_max, Pcl_max, P0_max, lr=5e-5, architecture="BB_small")
    
    elif architecture=="BB_large":
        mlmodel = qubitMLmodel(fb_max, tau_max, Pcl_max, P0_max, lr=5e-5, architecture="BB_large")
        
    elif architecture=="GB":
        mlmodel = qubitMLmodel(fb_max, tau_max, Pcl_max, P0_max, lr=5e-5, architecture="GB")
            
    mlmodel.load_model(model_name)
    mlmodel.maximums = maximums
    return mlmodel
###############################################################################    
def create_dataset(ratio):
    """
    This function reads the raw experimental data and postprocess it in a suitable format for machine learning, as well as
    spliiting into training and testing sets.
    
    ratio: integer representing percentage of training examples

    """
    dataset_x = []
    dataset_y = []
    with open("raw_dataset.csv") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            _,_,_, tau, temp, f_B, _,_,_, P0, P1, phi, Pcl = row
            dataset_x.append(np.array([[float(tau)*(1e6), float(f_B)*(1e-6), float(phi)*np.pi/180, float(temp), float(P0), float(P1) ]]))
            dataset_y.append(np.array([[float(Pcl)]]))
    dataset_x = np.concatenate(dataset_x, axis=0)
    dataset_y = np.concatenate(dataset_y, axis=0)

    num_training_ex = int(ratio*dataset_x.shape[0]/100)
    num_testing_ex  = dataset_x.shape[0] - num_training_ex
    
    idx_random = np.random.permutation(dataset_x.shape[0])
    training_x = np.concatenate( [dataset_x[idx:idx+1, :] for idx in idx_random[0:num_training_ex] ], 0)
    training_y = np.concatenate( [dataset_y[idx:idx+1, :] for idx in idx_random[0:num_training_ex] ], 0)
    
    testing_x  = np.concatenate( [dataset_x[idx:idx+1, :] for idx in idx_random[num_training_ex:(num_training_ex + num_testing_ex)] ], 0)
    testing_y  = np.concatenate( [dataset_y[idx:idx+1, :] for idx in idx_random[num_training_ex:(num_training_ex + num_testing_ex)] ], 0)

    dataset_sorted_x = {}
    dataset_sorted_y = {}
    for idx_ex in range(dataset_x.shape[0]):
        dataset_sorted_x["%f"%dataset_x[idx_ex, 1]] = [] 
        dataset_sorted_y["%f"%dataset_x[idx_ex, 1]] = []
    
    for idx_ex in range(dataset_x.shape[0]):
        dataset_sorted_x["%f"%dataset_x[idx_ex, 1]].append(dataset_x[idx_ex:idx_ex+1])
        dataset_sorted_y["%f"%dataset_x[idx_ex, 1]].append(dataset_y[idx_ex:idx_ex+1])
    
    for k in dataset_sorted_x.keys():
        dataset_sorted_x[k] = np.concatenate( dataset_sorted_x[k], axis=0 )
        dataset_sorted_y[k] = np.concatenate( dataset_sorted_y[k], axis=0 )
       
    l           = len(dataset_sorted_x.keys())
    idx_key     = np.random.permutation(l)
    sorted_keys = [list(dataset_sorted_x.keys())[idx] for idx in idx_key]
    dataset_x_sensing = [dataset_sorted_x[k] for k in sorted_keys]
    dataset_y_sensing = [dataset_sorted_y[k] for k in sorted_keys]

    with open("Dataset_April_2025_sorted_%d_%d"%(ratio, 100-ratio), "wb") as fl:
        [pickle.dump(arr, fl) for arr in [training_x, training_y, testing_x, testing_y, dataset_x_sensing, dataset_y_sensing, idx_random, idx_key]] 
        
    print(training_x.shape, training_y.shape, testing_x.shape, testing_y.shape, len(dataset_x_sensing), len(dataset_y_sensing))
###############################################################################
def lk_mlmodel_all(fb, parameters, model):
    """
    
    This function computes the likelihood using a trained ML model
    parameters: (n, 6): tau, fb, phi, T, P0, P1
    """
    
    tau_max, fb_max, phi_max, T_max, P0_max, Pcl_max = model.maximums
    testing_x = []
    for p in parameters:
        tau, _ , phi, T, P0, P1 = p
        testing_x.append( np.concatenate([np.ones((len(fb), 1))*tau/tau_max, np.expand_dims(fb,1)/fb_max, np.ones((len(fb), 1))*phi/phi_max, np.ones((len(fb), 1))*T/T_max, np.ones((len(fb), 1))*P0,np.ones((len(fb), 1))*P1], axis=1))
    
    testing_x = np.concatenate(testing_x, axis=0)
    
    pred = model.predict_measurements(testing_x, batch_size=len(fb))[:,0]
    
    return pred*Pcl_max
    
def bayes_procedure_all(F, prior, fb_true, rn, lk, perm):
    """
    This is the main Bayesian procedure
    
    F       : a list representing the detuning frequencies over which we compute the prior/posterior
    prior   : a list representing the prior at each point in F
    fb_true : the  true frequency needed for error analysis
    rn      : a list representing the measurements (number of "up" clicks in the batch if size R)
    lk      : the likelihood function computed over the interval F using the trained model
    perm    : a list of random (indices) to simualate random arrival of measurements

    """
    
    #1) calculate the prior statistics
    mse = []
    var = []
    est = []
    Pf  = prior/np.trapz(prior, F) # ensure normalization
    fest = np.trapz((F*Pf), F) 
    est.append(fest)
    mse.append((fest - fb_true)**2)
    var.append(np.trapz( ((F-fest)**2)*Pf, F) )
    posterior = [Pf]
    
    #2) for each new measurement, update prior to get posterior
    for n in range(rn.shape[0]): 
        
        # permute the measurement
        idx  = perm[n]
        
        # find the posterior utilizing the trained model
        P_cl = lk[(len(F)*idx):(len(F)*(idx+1))]
        
        Gauss_mu, Gauss_std = R*P_cl, rn[idx]*(R-rn[idx])/R 
        P_new = Pf*np.exp(-0.5*((rn[idx]-Gauss_mu)**2)/Gauss_std)/np.sqrt(2*np.pi*Gauss_std)
        
        # check that the new point is consistent
        if np.trapz(P_new, F)>0:
            Pf  = P_new  

        # normalize posterior
        Pf = Pf / np.trapz(Pf, F)
        posterior.append(Pf)
        # error analysis
        fest = np.trapz((F*Pf), F)
        est.append(fest)
        mse.append((fest - fb_true)**2)
        var.append(np.trapz( ((F-fest)**2)*Pf, F) )

    return est, mse, var, posterior
###############################################################################
def sensing_protocol(dataset_x_sensing, dataset_y_sensing, idx_key, F, prior, prior_name, model_WB, model_GB, model_BB, save=True, idx_permutations=None):
    """
    This function runs the Bayesian procedure over the dataset and computes statistics for performance analysis
    
    dataset_x_sensing: the inputs to the trained model: a dictionary of arrays of shape (n,6) representing (tau, fb, phi, T, P0, P1), where each entry corresponds to one fb
    dataset_y_sensing: the measured clicks formatted as dictionary  where each entry corresponds to one fb
    idx_key          : the index of which true frequency fb_true to process
    F                : a list representing the detuning frequencies over which we compute the prior/posterior
    prior            : a list representing the prior at each point in F
    prior_name       : a string to add to the filename of the files that saves the results
    model_WB         : a pretrained whitebox
    model_GB         : a pretrained graybox
    save             : Boolean, whether to export the figure or not for each example
    model_BB         : a pretrained blackbox
    idx_permutation  : a list of the permuted indices for each randomization of the Bayes procedure
    """
    
    n_Sensing        = dataset_x_sensing[idx_key].shape[0]
    fb_true          = dataset_x_sensing[idx_key][0, 1]
    rn               = dataset_y_sensing[idx_key]*R 
    num_permutations = 100
    lk_WB            = lk_mlmodel_all(F, dataset_x_sensing[idx_key], model_WB)
    lk_GB            = lk_mlmodel_all(F, dataset_x_sensing[idx_key], model_GB)
    lk_BB            = lk_mlmodel_all(F, dataset_x_sensing[idx_key], model_BB)
    
    est_WB  = []
    mse_WB  = []
    var_WB  = []
    
    est_GB  = []
    mse_GB  = []
    var_GB  = []
    
    est_BB  = []
    mse_BB  = []
    var_BB  = []
    
    posterior_WB = []
    posterior_GB = []
    posterior_BB = []
    
    if(idx_permutations is None):
        # generate random permutations for indexing the measurements
        idx_permutations = [np.random.permutation(n_Sensing) for _ in range(num_permutations)]
    
    # we repeat the Bayesian procedure multiple times and take statistics
    for repeation in range(num_permutations):
        x = bayes_procedure_all(F, prior, fb_true, rn, lk_WB, idx_permutations[repeation])
        est_WB.append(np.array(x[0]))
        mse_WB.append(np.array(x[1]))
        var_WB.append(np.array(x[2]))
        posterior_WB.append(x[3])

        x = bayes_procedure_all(F, prior, fb_true, rn, lk_GB, idx_permutations[repeation])
        est_GB.append(np.array(x[0]))
        mse_GB.append(np.array(x[1]))
        var_GB.append(np.array(x[2]))
        posterior_GB.append(x[3])
        
        x = bayes_procedure_all(F, prior, fb_true, rn, lk_BB, idx_permutations[repeation])
        est_BB.append(np.array(x[0]))
        mse_BB.append(np.array(x[1]))
        var_BB.append(np.array(x[2]))
        posterior_BB.append(x[3])
        
    # plots
    plt.figure(figsize=[12,5])
    plt.subplot(1,2,1)
    
    plt.loglog(range(1, n_Sensing+2), sum(mse_WB)/num_permutations, label="WB")
    plt.loglog(range(1, n_Sensing+2), sum(mse_GB)/num_permutations, label="GB")
    plt.loglog(range(1, n_Sensing+2), sum(mse_BB)/num_permutations, label="BB")
    plt.xlabel('Number of measurements')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid()
    
    plt.subplot(1,2,2)
    
    plt.plot(range(1, n_Sensing+2), sum(est_WB)/num_permutations, label="WB")
    plt.fill_between(range(1, n_Sensing+2), (sum(est_WB) - sum(np.sqrt(var_WB))*0.5)/num_permutations, (sum(est_WB) + sum(np.sqrt(var_WB))*0.5)/num_permutations, alpha=0.1)
    
    plt.plot(range(1, n_Sensing+2), sum(est_GB)/num_permutations, label="GB")
    plt.fill_between(range(1, n_Sensing+2), (sum(est_GB) - sum(np.sqrt(var_GB))*0.5)/num_permutations, (sum(est_GB) + sum(np.sqrt(var_GB))*0.5)/num_permutations, alpha=0.1)
    
    plt.plot(range(1, n_Sensing+2), sum(est_BB)/num_permutations, label="BB")
    plt.fill_between(range(1, n_Sensing+2), (sum(est_BB) - sum(np.sqrt(var_BB))*0.5)/num_permutations, (sum(est_BB) + sum(np.sqrt(var_BB))*0.5)/num_permutations, alpha=0.1)
    
    
    plt.axhline(fb_true, linestyle="--", color="black", label="True")
    plt.xlabel('Number of measurements')
    plt.ylabel(r'$f_b$ (MHz)')
    plt.legend(ncol=2)
    plt.grid()
    
    plt.suptitle("Bayesian estimation example %d, average over 100 permutations of the meaurement sequence"%idx_key)
    if save==True:
        plt.savefig("P%d.png"%idx_key, bbox_inches="tight")
    plt.show()
    
    # save the results
    np.savez("Results_%s_%d.npz"%(prior_name, idx_key), 
             est_WB  = np.concatenate([np.expand_dims(e, 0) for e in est_WB], 0),
             est_GB  = np.concatenate([np.expand_dims(e, 0) for e in est_GB], 0),
             est_BB  = np.concatenate([np.expand_dims(e, 0) for e in est_BB], 0),
             var_WB  = np.concatenate([np.expand_dims(e, 0) for e in var_WB], 0),
             var_GB  = np.concatenate([np.expand_dims(e, 0) for e in var_GB], 0),
             var_BB  = np.concatenate([np.expand_dims(e, 0) for e in var_BB], 0),
             mse_WB  = np.concatenate([np.expand_dims(e, 0) for e in mse_WB], 0),
             mse_GB  = np.concatenate([np.expand_dims(e, 0) for e in mse_GB], 0),
             mse_BB  = np.concatenate([np.expand_dims(e, 0) for e in mse_BB], 0),
             idx_permutations = np.array(idx_permutations)
             )
    
    fl = open("posterior_%s_%d.pckl"%(prior_name, idx_key), 'wb')
    pickle.dump(dict(posterior=[posterior_WB[0], posterior_GB[0], posterior_BB[0]]), fl, -1)
    fl.close()
###############################################################################
def sensing_protocol_noplot(dataset_x_sensing, dataset_y_sensing, idx_key, F, prior, prior_name, model_WB, model_GB, model_BB, idx_permutations):
    """
    This function runs the Bayesian procedure over the dataset and computes statistics for performance analysis but not plots
    
    dataset_x_sensing: the inputs to the trained model: a dictionary of arrays of shape (n,6) representing (tau, fb, phi, T, P0, P1), where each entry corresponds to one fb
    dataset_y_sensing: the measured clicks formatted as dictionary  where each entry corresponds to one fb
    idx_key          : the index of which true frequency fb_true to process
    F                : a list representing the detuning frequencies over which we compute the prior/posterior
    prior            : a list representing the prior at each point in F
    prior_name       : a string to add to the filename of the files that saves the results
    model_WB         : a pretrained whitebox
    model_GB         : a pretrained graybox
    model_BB         : a pretrained blackbox
    idx_permutation  : a list of the permuted indices for each randomization of the Bayes procedure
    """
    n_Sensing        = dataset_x_sensing[idx_key].shape[0]
    fb_true          = dataset_x_sensing[idx_key][0, 1]
    rn               = dataset_y_sensing[idx_key]*R  
    num_permutations = 100
    lk_WB            = lk_mlmodel_all(F, dataset_x_sensing[idx_key], model_WB)
    lk_GB            = lk_mlmodel_all(F, dataset_x_sensing[idx_key], model_GB)
    lk_BB            = lk_mlmodel_all(F, dataset_x_sensing[idx_key], model_BB)

    est_WB  = []
    mse_WB  = []
    var_WB  = []
    
    est_GB  = []
    mse_GB  = []
    var_GB  = []
    
    est_BB  = []
    mse_BB  = []
    var_BB  = []
    
    posterior_WB = []
    posterior_GB = []
    posterior_BB = []
    
    if(idx_permutations is None):
        # generate random permutations for indexing the measurements
        idx_permutations = [np.random.permutation(n_Sensing) for _ in range(num_permutations)]
        
    # we repeat the Bayesian procedure multiple times and take statistics
    for repeation in range(num_permutations):
        x = bayes_procedure_all(F, prior, fb_true, rn, lk_WB, idx_permutations[repeation])
        est_WB.append(np.array(x[0]))
        mse_WB.append(np.array(x[1]))
        var_WB.append(np.array(x[2]))
        posterior_BB.append(x[3])
        
        x = bayes_procedure_all(F, prior, fb_true, rn, lk_GB, idx_permutations[repeation])
        est_GB.append(np.array(x[0]))
        mse_GB.append(np.array(x[1]))
        var_GB.append(np.array(x[2]))
        posterior_BB.append(x[3])
        
        x = bayes_procedure_all(F, prior, fb_true, rn, lk_BB, idx_permutations[repeation])
        est_BB.append(np.array(x[0]))
        mse_BB.append(np.array(x[1]))
        var_BB.append(np.array(x[2]))
        posterior_BB.append(x[3])
        
    # save the results 
    np.savez("Results_%s_%d.npz"%(prior_name, idx_key), 
             est_WB  = np.concatenate([np.expand_dims(e, 0) for e in est_WB], 0),
             est_GB  = np.concatenate([np.expand_dims(e, 0) for e in est_GB], 0),
             est_BB  = np.concatenate([np.expand_dims(e, 0) for e in est_BB], 0),
             var_WB  = np.concatenate([np.expand_dims(e, 0) for e in var_WB], 0),
             var_GB  = np.concatenate([np.expand_dims(e, 0) for e in var_GB], 0),
             var_BB  = np.concatenate([np.expand_dims(e, 0) for e in var_BB], 0),
             mse_WB  = np.concatenate([np.expand_dims(e, 0) for e in mse_WB], 0),
             mse_GB  = np.concatenate([np.expand_dims(e, 0) for e in mse_GB], 0),
             mse_BB  = np.concatenate([np.expand_dims(e, 0) for e in mse_BB], 0),
             idx_permutations = np.array(idx_permutations)
             )
    
    fl = open("posterior_%s_%d.pckl"%(prior_name, idx_key), 'wb')
    pickle.dump(dict(posterior=[posterior_WB[0], posterior_GB[0], posterior_BB[0]]), fl, -1)
    fl.close()
###############################################################################    