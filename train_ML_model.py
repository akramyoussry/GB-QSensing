import numpy as np
import sys, pickle
from QubitMLModel import qubitMLmodel
####################################################################
num_iter = 100000
####################################################################
def verify(ex):
    tau, fb, phi, _, P0, P1 = ex
    tau   = tau * tau_max
    fb    = fb * fb_max
    phi   = phi * 2*np.pi
    alpha = 0.5*(P0 + P1)
    V     = (P0 - P1)*np.exp(-(tau/T2)**2)/(P0+P1)
    Pcl   = alpha*(1 + V*np.cos(2*np.pi*fb*tau + phi) )
    return Pcl/Pcl_max
#####################################################################
if __name__ == '__main__':
    
    dataset = sys.argv[1]
    box     = sys.argv[2]
    
    print(dataset, box)
    
    #1) Load Dataset
    with open("Dataset_April_2025_sorted_90_10", "rb") as fl:
        training_x, training_y, testing_x, testing_y, testing_x_sensing, testing_y_sensing = [pickle.load(fl) for _ in range(6) ]

    #2) Normalize the parameters for the ML 
    
    tau_max = np.max([np.max(training_x[:,0]), np.max(testing_x[:,0])])
    fb_max  = np.max([np.max(np.abs(training_x[:,1])), np.max(np.abs(testing_x[:,1]))])
    phi_max = 2*np.pi
    T_max  = np.max([np.max(training_x[:,3]), np.max(testing_x[:,3])])
    P0_max = np.max([np.max(training_x[:,4:]), np.max(testing_x[:,4:])])
    
    training_x = np.concatenate([training_x[:, 0:1]/tau_max, training_x[:, 1:2]/fb_max, training_x[:, 2:3]/phi_max, training_x[:, 3:4]/T_max,  training_x[:, 4:] ], axis=-1)
    testing_x  = np.concatenate([testing_x[:,  0:1]/tau_max,  testing_x[:, 1:2]/fb_max,  testing_x[:, 2:3]/phi_max,  testing_x[:, 3:4]/T_max,   testing_x[:, 4:] ], axis=-1)   
    
    Pcl_max    = np.max([np.max(training_y), np.max(testing_y)])
    training_y = training_y/Pcl_max
    testing_y  = testing_y/Pcl_max
        
    batch_size = training_x.shape[0]
    T2    = 0.5*tau_max
    
    if dataset=="sim":
      training_y = np.concatenate([np.reshape(verify(ex), (1,1)) for ex in training_x])
      testing_y  = np.concatenate([np.reshape(verify(ex), (1,1)) for ex in testing_x])
#######################################################################
    if box=="WB":
        mlmodel = qubitMLmodel(fb_max, tau_max, Pcl_max, P0_max, architecture="WB")
        mlmodel.model.set_weights([116.0])
        print(mlmodel.model.get_weights())
        mlmodel.train_model_val(training_x, training_y, testing_x, testing_y, num_iter, batch_size)
        print(mlmodel.model.get_weights())
    
    elif box=="BB_small":
        mlmodel = qubitMLmodel(fb_max, tau_max, Pcl_max, P0_max, lr=1e-4, architecture="BB_small")
    
    elif box=="BB_large":
        mlmodel = qubitMLmodel(fb_max, tau_max, Pcl_max, P0_max, lr=5e-5, architecture="BB_large")
        
    elif box=="GB":
        mlmodel = qubitMLmodel(fb_max, tau_max, Pcl_max, P0_max, lr=5e-5, architecture="GB")
            
        
    mlmodel.train_model_val(training_x, training_y, testing_x, testing_y, num_iter, batch_size)
    
    mlmodel.save_model("model_%s_%s_12_08_2026"%(box,dataset))
    
    print(mlmodel.training_history[-1])
    print(mlmodel.val_history[-1])