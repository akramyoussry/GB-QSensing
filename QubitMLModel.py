"""
This module implements the machine learning-based model for the qubit. It has three classes:
    VoConstruction         : This is an internal class for constructing the Vo operators
    Pcl_WB                 : This is an internal class for computing click probabilities for the whitebox
    RamseyUnitary          : This is an internal class for constructing the equivalent unitary of a Ramsey experiment
    QuantumMeasurement     : This is an internal class to model coupling losses at the output.
    qubitMLmodel           : This is the main class that defines machine learning model for the qubit.  
"""

# Preamble
import numpy as np
from scipy import linalg
import tensorflow as tf
from tensorflow.keras import layers,optimizers,Model,initializers,constraints
import pickle
###############################################################################
class VoConstruction(layers.Layer):
    """
    This class defines a custom tensorflow layer that takes a vector of parameters represneting eigendecompostion and reconstructs the Vo operator
    """
    
    def __init__(self, invO,  **kwargs):
        """
        Class constructor
        
        O: The inverse of the observable to be measaured
        """
        # this has to be called for any tensorflow custom layer
        super(VoConstruction, self).__init__(**kwargs)
    
        self.invO = tf.constant(invO, dtype=tf.complex128)
        
    def call(self, x):
        """
        This method must be defined for any custom layer, it is where the calculations are done.   
        
        x: a tensor representing the inputs to the layer. This is passed automatically by tensorflow. 
        """ 
        
        # retrieve the two types of parameters from the input: 3 eigenvector parameters and 2 eigenvalue parameters
        U,x1,x2 = x
        
        # parametrize eigenvector matrix being unitary as in https://en.wikipedia.org/wiki/Unitary_matrix 
        psi   = tf.cast( U[:,0:1], tf.complex128)*1j
        theta = U[:,1:2]
        delta = tf.cast( U[:,2:], tf.complex128)*1j 
        
        # construct the first matrix
        A = tf.linalg.diag(tf.concat([tf.exp(psi), tf.exp(-psi)], -1))
        
        # construct the second matrix
        B1 = tf.expand_dims( tf.concat([tf.cos(theta), tf.sin(-theta)],-1), -1)
        B2 = tf.expand_dims( tf.concat([tf.sin(theta), tf.cos(theta)],-1), -1)
        
        B  = tf.cast( tf.concat([B1,B2],-1), tf.complex128) 
        
        # construct the third matrix
        C = tf.linalg.diag(tf.concat([tf.exp(delta), tf.exp(-delta)], -1))
        
        # multiply all three to get a Unitary (global phase shift is neglected)
        U = tf.matmul(A, tf.matmul(B,C) )
        
        # construct eigenvalue matrix such that it is traceless
        lambda1 = x1
        lambda2 = tf.multiply(x2 , 1-tf.abs(x1) )
        d = tf.concat([lambda1, lambda2], -1)*2
        d = tf.cast( tf.linalg.diag(d), tf.complex128)
        
        # construct the Hermitian tracelesss operator from its eigendecompostion
        H = tf.matmul( tf.matmul(U, d), U, adjoint_b=True)    
        
        # expand the observable operator along batch axis
        invO = tf.expand_dims(self.invO, 0)
        temp_shape = tf.concat( [tf.shape(U)[0:1], tf.constant(np.array([1,1],dtype=np.int32))], 0 )
        invO = tf.tile(invO, temp_shape)
        
        # Construct Vo operator        
        return tf.matmul(invO, H)   
##############################################################################
class Pcl_WB(layers.Layer):
    """
    This class defines a custom layer that calculates the click probability accroding to Gaussian profile.
    
    """
    def __init__(self, fb_max, tau_max, **kwargs):
        """
        class constuctructor
        
        fb_max:  the maximum of the dataset to unormalize the detuning frequency
        tau_max: the maximum of the dataset to unormlaize the time delay
        """
        self.fb_max  = fb_max
        self.tau_max = tau_max
        
        super(Pcl_WB, self).__init__(**kwargs)
        
        self.T2_star_inv  = self.add_weight(name="T2_star_inv", dtype=tf.float32, trainable=True, initializer = initializers.RandomNormal(mean=1/(0.1*tau_max), stddev=20e3), constraint=constraints.NonNeg())
    
    def call(self, x):
        """
        The calculation
        """
        # extract pulse parameters and unnormalize them
        tau = x[:,0:1]*self.tau_max
        fb  = x[:,1:2]*self.fb_max
        phi = -x[:,2:3]*2*np.pi
        P0  = x[:,3:4]
        P1  = x[:,4:5]
        
        alpha = (P0+P1)*0.5
        V     = tf.exp(-tf.square(tau*self.T2_star_inv)) * (P0 - P1)/(P0 + P1)
        Pcl   = alpha * (1 + V*tf.cos(2*np.pi*fb*tau + phi))
        return Pcl
##############################################################################
class RamseyUnitary(layers.Layer):
    """
    This class defines a custom layer that takes the pulse sequence parameters and generates the ideal Ramsey Unitary
    
    """
    
    def __init__(self, fb_max, tau_max, **kwargs):
        """
        class constructor
        
        f_max:   the maximum of the dataset to unormalize the detuning frequency
        tau_max: the maximum of the dataset to unormlaize the time delay
        """
        self.fb_max   = fb_max
        self.tau_max  = tau_max
        self.sqrtX    = linalg.expm(-1j*0.5*np.pi*np.array([[0,1],[1,0]])/2)
        self.sqrtX    = tf.expand_dims( tf.constant(self.sqrtX, dtype=tf.complex128)/np.sqrt(2), 0 )
        self.Z        = tf.expand_dims(tf.constant([[1,0],[0,-1]], tf.complex128), 0)
        
        super(RamseyUnitary, self).__init__(**kwargs)
        
    def call(self, x):
        """
        This method must be defined for any custom layer, it is where the calculations are done.   
   
        x: a tensor representing the inputs to the layer. This is passed automatically by tensorflow.  
        """
        # extract pulse parameters and unnormalize them
        tau = x[:,0:1]*self.tau_max
        fb  = x[:,1:2]*self.fb_max
        phi = x[:,2:3]*2*np.pi
        
        # calculate total acquired phase shift due to free evolution and phase shift 
        theta      = tf.expand_dims( tf.multiply(2*np.pi*fb, tau) + phi, -1)
        temp_shape = tf.constant(np.array([1,2,2], dtype=np.int32))
        theta      = tf.cast(tf.tile(theta, temp_shape), tf.complex128)
        
        # construct the ideal pulse unitaries
        temp_shape = tf.concat( [tf.shape(x)[0:1], tf.constant(np.array([1,1],dtype=np.int32))],0 )
        sqrtX = tf.tile(self.sqrtX, temp_shape)
        Z     = tf.tile(self.Z, temp_shape)
        
        # compute the total unitary
        U = tf.matmul(sqrtX, tf.matmul( tf.linalg.expm(-1j*tf.multiply(theta, Z)*0.5), sqrtX) )
        return U       
###############################################################################    
class QuantumMeasurement(layers.Layer):
    """
    This class defines a custom tensorflow layer that takes the unitary as input, 
    and generates the measurement outcome probability as output
    """
    
    def __init__(self, initial_state, measurement_operator, **kwargs):
        """
        Class constructor
        
        initial_state       : The inital density matrix of the state before evolution.
        Measurement_operator: The measurement operator
        """          
        self.initial_state        = tf.constant(initial_state, dtype=tf.complex128)
        self.measurement_operator = tf.constant(measurement_operator, dtype=tf.complex128)
    
        # we must call thus function for any tensorflow custom layer
        super(QuantumMeasurement, self).__init__(**kwargs)
            
    def call(self, x): 
        """
        This method must be defined for any custom layer, it is where the calculations are done.   
        
        x: a tensor representing the inputs to the layer. This is passed automatically by tensorflow. 
        """ 
    
        # extract the different inputs of this layer which are the Vo and Uc
        Vo, Uc = x
        
        # construct a tensor in the form of a row vector whose elements are [d1,1,1], where d1 correspond to the number of examples of the input
        temp_shape = tf.concat( [tf.shape(Uc)[0:1],tf.constant(np.array([1,1],dtype=np.int32))],0 )

        # add an extra dimension for the initial state and measurement tensors to represent batch
        initial_state        = tf.expand_dims(self.initial_state,0)
        measurement_operator = tf.expand_dims(self.measurement_operator,0)   
        
        # repeat the initial state and measurment tensors along the batch dimensions
        initial_state        = tf.tile(initial_state, temp_shape )
        measurement_operator = tf.tile(measurement_operator, temp_shape)   
        
        # evolve the initial state using the propagator provided as input
        final_state = tf.matmul(tf.matmul(Uc, initial_state), Uc, adjoint_b=True )
        
        # calculate the expectation value
        expectation = tf.linalg.trace( tf.matmul( tf.matmul( Vo, final_state), measurement_operator) ) 
        
        return tf.squeeze( tf.reshape( tf.math.real(expectation), temp_shape), axis=-1 )    
###############################################################################    
class qubitMLmodel():
    """
    This is the main class that defines machine learning model of the qubit.
    """    
    def __init__(self, fb_max, tau_max, Pcl_max, P0_max, lr=0.01, architecture="WB"):
        """
        Class constructor.
        
        fb_max : maximum detuning in the dataset for normalization
        tau_max: maximum delay in the dataset for normalization
        Pcl_max: maximum click probability (output) in the datasrt for normalization
        lr     : learning rate for training
        architecture   : one of "WB", "GB", "BB_small" or ""BB_large"" 
        
        """
              
        # order of inputs tau, f_b, phi, T, P0, P1
        self.architecture      = architecture
        self.training_history  = []
        self.val_history       = []
        self.P0_max            = P0_max
        if architecture=="WB":
            
            num_inputs_WB = 5 #[tau, fb, phi, P0, P1]
            
            # define a tensorflow input layer for the whitebox calculation
            pulse_parameters_WB = layers.Input(shape=(num_inputs_WB, ), name="Pulse_parameters_WB")
            
            # whitebox calculations           
            Pcl     = Pcl_WB(fb_max, tau_max)(pulse_parameters_WB)
            
            Pcl_scaled = layers.Lambda(lambda x: x/Pcl_max)(Pcl)
            # define the Keras model
            self.model = Model(pulse_parameters_WB, Pcl_scaled)
            
            # specify the optimizer and loss function for training 
            self.model.compile(optimizer=optimizers.Adam(learning_rate=lr), loss='mean_squared_logarithmic_error')
            
            # print a summary of the model showing the layers, their connections, and the number of training parameters
            self.model.summary()  
  
        #######################################################################################################
        elif architecture=="GB":
            # initialize and store class parameters
            Z                      = np.array([[1,0],[0,-1]])
            rho                    = np.array([[1,0],[0,0]]) 
            num_inputs_BB          = 3 
            num_inputs_WB          = 5
            
            # define a tensorflow input layer for the normalized pulse sequence parameters
            pulse_parameters_BB = layers.Input(shape=(num_inputs_BB, ), name="Pulse_parameters_BB")
        
            # define a second tensorflow input layer for the whitebox calculation
            pulse_parameters_WB = layers.Input(shape=(num_inputs_WB, ), name="Pulse_parameters_WB")
        
            # define an NN taking pulse sequence parameters
            params = layers.Dense(1024, activation="tanh")(pulse_parameters_BB) 
            for num_nodes in [512,128,64,32,16,8,4]:
                params = layers.Dense(num_nodes, activation="tanh")(params)
            
            # define two NNs one for the producing the eigenvector parameters of the Vo operator and another one for the eigenvalues, and repeat for each Vo operator
            Vo = VoConstruction(invO = Z, name="VZ")(
                    [layers.Dense(3, activation='linear')(params),  layers.Dense(1, activation='tanh')(params), layers.Dense(1, activation='tanh')(params)])
                   
    
            # define the custom defined tensorflow layer that constructs the final control propagator
            Unitary     = RamseyUnitary(fb_max, tau_max, name="Unitary")(pulse_parameters_WB)
            
            # add the custom defined tensorflow layer that calculates the measurement outcomes
            expectations = QuantumMeasurement(rho,Z, name="Expectation")([Vo,Unitary])
           
            # add custom layer for readout calculation
            Pcl = layers.Lambda(lambda x: 0.5*x[0][:,3:4]*(1+x[1]) + 0.5*x[0][:,4:]*(1-x[1]), name="Pclick")([pulse_parameters_WB, expectations])
            
            Pcl_scaled = layers.Lambda(lambda x: x/Pcl_max)(Pcl)
            # define now the tensorflow model
            self.model    = Model( inputs = [pulse_parameters_BB, pulse_parameters_WB], outputs = Pcl_scaled )
            
            # specify the optimizer and loss function for training 
            self.model.compile(optimizer=optimizers.Adam(learning_rate=lr), loss='mean_squared_logarithmic_error')
            
            # print a summary of the model showing the layers, their connections, and the number of training parameters
            self.model.summary()
       #######################################################################################################
        elif architecture=="BB_small":

            num_inputs_BB     = 5 
            
            # define a tensorflow input layer for the normalized pulse sequence parameters
            pulse_parameters_BB = layers.Input(shape=(num_inputs_BB, ), name="Pulse_parameters_BB")
            
            # define an NN taking pulse sequence parameters
            params = layers.Dense(1024, activation="relu")(pulse_parameters_BB) 

            # add hidden layers
            for num_nodes in [512,128,64,32,16,8,4,4]:
                params = layers.Dense(num_nodes, activation="relu")(params)
            
            # add output layer
            Pcl = layers.Dense(1, activation ="sigmoid")(params)
            
            Pcl_scaled = layers.Lambda(lambda x: x/Pcl_max)(Pcl)
            
            # define now the tensorflow model
            self.model    = Model( inputs = pulse_parameters_BB, outputs = Pcl_scaled )
            
            # specify the optimizer and loss function for training 
            self.model.compile(optimizer=optimizers.Adam(learning_rate=lr), loss='mean_squared_logarithmic_error')
            
            # print a summary of the model showing the layers, their connections, and the number of training parameters
            self.model.summary()
        ######################################################################################################
        elif architecture=="BB_large":

            num_inputs_BB     = 5 
            
            # define a tensorflow input layer for the normalized pulse sequence parameters
            pulse_parameters_BB = layers.Input(shape=(num_inputs_BB, ), name="Pulse_parameters_BB")
            
            # define an NN taking pulse sequence parameters
            params = layers.Dense(16, activation="relu")(pulse_parameters_BB) 

            for num_nodes in [32,64,128,512,1024,2048,4096,4096,2048,1024,512,128,64,32,16,8,4]:
                params = layers.Dense(num_nodes, activation="relu")(params)
            
            # add hidden layers
            Pcl = layers.Dense(1, activation ="sigmoid")(params)
            
            # add output layer
            Pcl_scaled = layers.Lambda(lambda x: x/Pcl_max)(Pcl)
            
            # define now the tensorflow model
            self.model    = Model( inputs = pulse_parameters_BB, outputs = Pcl_scaled )
            
            # specify the optimizer and loss function for training 
            self.model.compile(optimizer=optimizers.Adam(learning_rate=lr), loss='mean_squared_logarithmic_error')
            
            # print a summary of the model showing the layers, their connections, and the number of training parameters
            self.model.summary()
        ######################################################################################################
        else:
            raise Exception("Unknown model architecture")            
    
    def train_model_val(self, training_x, training_y, testing_x, testing_y, epochs, batch_size):
        """
        This method is for training the model given the training set and the validation set
        
        training_x: A (number of examples, 6) with parameter order tau, fb, phi, T, P0, P1
        training_y: A numpy array that stores the meeasurement outcomes (number of examples,1).
        epochs    : The number of iterations to do the training  
        batch_size: Batch size
        """        
        if self.architecture=="WB":
            WB_training_data = np.concatenate([training_x[:, 0:3],training_x[:, 4:]], axis=-1)
            WB_testing_data  = np.concatenate([testing_x[:, 0:3], testing_x[:, 4:]], axis=-1)
            # Train the model for "epochs" number of iterations using the provided training set, and store the training history
            h  =  self.model.fit(WB_training_data, training_y, epochs=epochs, batch_size=batch_size,verbose=2,validation_data = (WB_testing_data, testing_y)) 

        elif self.architecture=="GB":         
            WB_training_data = np.concatenate([training_x[:, 0:3],training_x[:, 4:]], axis=-1)
            WB_testing_data  = np.concatenate([testing_x[:, 0:3], testing_x[:, 4:]], axis=-1)


            BB_training_data = training_x[:, 0:3]
            BB_testing_data  = testing_x[:, 0:3]
             
            # Train the model for "epochs" number of iterations using the provided training set, and store the training history
            h  =  self.model.fit([BB_training_data, WB_training_data], training_y, epochs=epochs, batch_size=batch_size,verbose=2,validation_data = ([BB_testing_data, WB_testing_data], testing_y)) 
         
        elif self.architecture=="BB_small" or self.architecture=="BB_large":

            BB_training_data = np.concatenate([training_x[:, 0:3],training_x[:, 4:]/self.P0_max], axis=-1)
            BB_testing_data  = np.concatenate([testing_x[:, 0:3],  testing_x[:, 4:]/self.P0_max], axis=-1)
            
            # Train the model for "epochs" number of iterations using the provided training set, and store the training history
            h  =  self.model.fit(BB_training_data, training_y, epochs=epochs, batch_size=batch_size,verbose=2,validation_data = (BB_testing_data, testing_y)) 
            

        self.training_history  = h.history["loss"]
        self.val_history       = h.history["val_loss"]
               
    def predict_measurements(self, testing_x, batch_size):
        """
        This method is for predicting the measurement outcomes using the trained model. Usually called after training.
        
        testing_x: A list of two  numpy arrays the first is of shape (number of examples, number of signal parameters), and the second is of dimensions (number of examples, number of time steps, 1)
        """   
        
        if self.architecture=="WB":
            WB_testing_data  = np.concatenate([testing_x[:, 0:3], testing_x[:, 4:]], axis=-1)
            return self.model.predict(WB_testing_data, batch_size=batch_size)
        
        elif self.architecture=="GB":
            WB_testing_data  = np.concatenate([testing_x[:, 0:3], testing_x[:, 4:]], axis=-1)
            BB_testing_data  = testing_x[:, 0:3]
                
            return self.model.predict([BB_testing_data, WB_testing_data], batch_size=batch_size)
        
        elif self.architecture=="BB_small" or self.architecture=="BB_large":

            BB_testing_data  = np.concatenate([testing_x[:, 0:3], testing_x[:, 4:]/self.P0_max], axis=-1)
            return self.model.predict(BB_testing_data, batch_size=batch_size)
        
    def predict_control_unitary(self,testing_x):
        """
        This method is for evaluating the control unitary. Usually called after training.
        
        testing_x: A list of two  numpy arrays the first is of shape (number of examples, number of signal parameters), and the second is of dimensions (number of examples, number of time steps, 1)
        """
        if self.architecture=="GB":
         
            WB_testing_data  = np.concatenate([testing_x[:, 0:3], testing_x[:, 4:]], axis=-1)
            BB_testing_data  = testing_x[:, 0:3]
    
            # define a new model that connects the input voltage and the GRU output 
            unitary_model = Model(inputs=self.model.input, outputs=self.model.get_layer('Unitary').output)
        
            # evaluate the output of this model
            return unitary_model.predict([BB_testing_data, WB_testing_data])            
        else:
            raise Exception("Wrong archictecture to predict Uctrl")
              
    def predict_Vo(self, testing_x):
        """
        This method is for predicting the Vo operators. Usally called after training.
       
        testing_x: A list of two  numpy arrays the first is of shape (number of examples, number of signal parameters), and the second is of dimensions (number of examples, number of time steps, 1)
        """
        if (self.architecture=="GB"):       
            WB_testing_data  = np.concatenate([testing_x[:, 0:3], testing_x[:, 5:]], axis=-1)
            BB_testing_data  = testing_x[:, 0:3]
                         
            # define a new model that connects the inputs to each of the Vo output layers
            Vo_model = Model(inputs=self.model.input, outputs=[self.model.get_layer(V).output for V in ["V0","V1","V2"] ] )
            
            # predict the output of the truncated model which is VO. 
            Vo = Vo_model.predict([BB_testing_data,WB_testing_data])
          
            return Vo
        else:
            raise Exception("Wrong archictecture to predict Vo")
              
    def save_model(self, filename):
        """
        This method is to export the model to an external .mlmodel file
        
        filename: The name of the file (without any extensions) that stores the model.
        """
                
        data = {'training_history':self.training_history, 
                 'val_history'    :self.val_history,
                 'model_weights'  :self.model.get_weights(),
                }
        f = open(filename, 'wb')
        pickle.dump(data, f, -1)
        f.close()

    def load_model(self, filename):
        """
        This method is to import the models from an external .mlmodel file
        
        filename: The name of the file (without any extensions) that stores the model.
        """       
        f = open(filename, 'rb')
        data = pickle.load(f)
        f.close()          
        self.training_history  = data['training_history']
        self.val_history       = data['val_history']
        self.model.set_weights(data['model_weights'])
###############################################################################