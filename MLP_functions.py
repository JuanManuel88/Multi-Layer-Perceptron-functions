import numpy as np

def activation_function(z, activation, slope):
    """
    Returns the selected activation function and its gradient
    """
    if activation == "sigmoid":
        a  = np.reciprocal(1 + np.exp(-z))
        Da = np.multiply(a, 1 - a)
    
    elif activation == "softmax":
        a  = np.exp(z)/np.sum(np.exp(z), axis=0)
        Da = np.multiply(a, 1 - a)   #given that only dai backpropagates to ni, only dzi=dai.*g(zi)' is useful
    
    elif activation == "tanh":
        a = np.divide(np.exp(z) - np.exp(-z), np.exp(z) + np.exp(-z))
        Da = 1-a**2
    
    elif activation == "relu":
        a = np.where(z >= 0, z, 0)
        Da = np.where(z >= 0, 1, 0)
    
    elif activation == "leaky_relu":
        a = np.where(z >= slope*z, z, slope*z)
        Da = np.where(z >=0, 1, slope)
    else:
        print("Error: activation argument must be a sigmoid, softmax, tanh, relu, or leaky_relu string")
        
    return a, Da

def parameters_initialization(input_data, MLP_struct):
    """
    Arguments:
    MLP_struct: number of neurons in each layer, hidden and output
    data_input: matrix where every column is an input, with shape (k, m)
    
    Returns:
    A list of parameters for each layer, where the last column is the bias (+1)
    """
    
    k = input_data.shape[0]
    MLP_struct.insert(0, k)
    L = len(MLP_struct) #number of layer including the output and input layer
    list_parameters = []
    
    #for each layer
    for l in range(1, L):
        parameters = np.random.randn(MLP_struct[l], MLP_struct[l-1]+1)*0.01
        list_parameters.append(parameters)

    return list_parameters

def xavier_he_initialization(input_data, MLP_struct, hidden_activation):
    """
    Arguments:
    MLP_struct: number of neurons in each layer, hidden and output
    data_input: matrix where every column is an input, with shape (k, m)
    
    Returns:
    A list of parameters for each layer, where the last column is the bias (+1)
    """
    
    k = input_data.shape[0]
    MLP_struct.insert(0, k)
    L = len(MLP_struct)                         #number of layer including the output and input layer
    MLP_struct.append(MLP_struct[-1])
    list_parameters = []
    
    #xavier
    if hidden_activation == "tanh":
        
        #from the first hidden layer (l=1) up to the output layer (l=L-1)    
        for l in range(1, L):                   
            a = (6/(MLP_struct[l]+MLP_struct[l+1]))**0.5
            parameters = np.random.uniform(low = -a, high = a, size = (MLP_struct[l], MLP_struct[l-1]+1))
            list_parameters.append(parameters)     

    #he
    elif hidden_activation == "relu":
    
        #from the first hidden layer (l=1) up to the output layer (l=L-1)      
        for l in range(1, L):
            var = 2/(MLP_struct[l]/2 + MLP_struct[l+1]/2)
            parameters = np.random.randn(MLP_struct[l], MLP_struct[l-1]+1)*np.sqrt(var)
            list_parameters.append(parameters)
    else:
        
        #from the first hidden layer (l=1) up to the output layer (l=L-1)      
        for l in range(1, L):
            parameters = np.random.randn(MLP_struct[l], MLP_struct[l-1]+1)*0.01
            list_parameters.append(parameters)

    return list_parameters

def feedforward(input_data, parameters, hidden_activation, output_activation, slope):
    """
    Arguments:
    data_input: matrix where every column is an input, with shape (k, m)
    parameters: list of weight and bias matrixes for each layer
    activations: string of the activation function name
    
    Returns:
    list_A: Activations of each neuron of each layer, including the input data. len(list_A) = L + 1
    list_Z: Linear output of each neuron of each layer. len(list_Z) = L
    list_dAdZ: Derivatives of each activation wrt Z. len(list_dAdZ) = L
    """
    
    L = len(parameters)
    list_Z = []
    list_A = []
    list_dAdZ = []
    list_A.append(input_data)
    
    for l in range(L-1):
        Z       = np.dot(parameters[l][:, 0:-1], list_A[l]) + parameters[l][:, -1].reshape((-1,1))
        A, dAdZ = activation_function(Z, hidden_activation, slope)
        list_Z.append(Z)
        list_A.append(A)
        list_dAdZ.append(dAdZ)
    
    #for the output layer (l = L-1):
    Z_out               = np.dot(parameters[-1][:, 0:-1], list_A[-1]) + parameters[-1][:, -1].reshape((-1,1))
    A_out, dA_outdZ_out = activation_function(Z_out, output_activation, slope)
    list_Z.append(Z_out)
    list_A.append(A_out)
    list_dAdZ.append(dA_outdZ_out)
    
    return list_A, list_Z, list_dAdZ

def Je_function(A_out, Target, classification):
    """
    Arguments:
    A_out: final activation of the MLP
    Target: Target values
    classification: binary or multiclass
    
    Returns
    Je: the sum of the loss function over N outputs, without regularization, for the MLP 
    dA_out: gradient of loss function w.r.t. A_out
    """
    N = A_out.shape[1]
    
    if classification == "binary":
        Je = (-1/N)*(np.dot(Target, np.log(A_out.T)) + np.dot((1 - Target), np.log(1 - A_out.T)))
        dA_out = np.divide(A_out - Target, np.multiply(A_out, 1 - A_out))
        
    elif classification == "multiclass":
        Je = (-1/N)*np.sum(np.multiply(Target, np.log(A_out)))
        dA_out = np.divide(A_out - Target, np.multiply(A_out, 1 - A_out))
    
    else:
        print("Error: classification argument must only be a binary or multiclass string")

    return Je, dA_out

def backpropagation(dA_out, list_dAdZ, list_A, parameters):
    """
    Arguments:
    dA
    list_dAdZ : list containing the derivations of the activation w.r.t. the linear combination of the inputs
    list_A    : list containing the derivations of the activation w.r.t. the linear combination of the inputs
    parameters: list of weight and bias matrixes for each layer
    
    Returns:
    List_gradients: Gradients of each parameter of each neuron
    list_dZ: Gradient of the loss function wrt Z
    list_dA: Gradient of the loss function wrt A
    """
    N = dA_out.shape[1] #number of outputs
    L = len(parameters)
    
    list_dZ = []
    list_gradients = []
    list_dA = []
    list_dA.append(dA_out)
    
    for l in reversed(range(L)):
        dZ = np.multiply(list_dA[L-1-l], list_dAdZ[l])
        dW = (1/N)*np.dot(dZ, list_A[l].T)  #A[l] is the activation of the layer "l-1"
        db = (1/N)*np.sum(dZ, axis = 1, keepdims = True)
        
        dA_prev = np.dot(parameters[l][:, 0:-1].T, dZ)
        
        list_dZ.append(dZ)
        list_gradients.append(np.concatenate((dW, db), axis=1))
        list_dA.append(dA_prev)
        
    list_dZ.reverse()
    list_gradients.reverse()
    list_dA.reverse()
    
    return list_gradients, list_dZ, list_dA

def adam_optimization(M, V, beta1, beta2, t, parameters, alpha, gradients):
    """
    M and V must have the same dimensions as the gradients in each layer
    beta1 and beta2 are usually 0.9 and 0.999 (from paper)
    alpha is usually 0.001 (from paper)
    t: number of iteration step, starting from 1
    """
    
    L = len(parameters)
    
    list_new_parameters = []
    Mnew = []
    Vnew = []
    
    for l in range(L):
        Mnewl = beta1*M[l] + (1 - beta1)*gradients[l]
        Vnewl = beta2*V[l] + (1 - beta2)*(gradients[l]**2)
        
        Mcorrected = Mnewl/(1 - beta1**t)
        Vcorrected = Vnewl/(1 - beta2**t)
        eps = 10**-8   #from paper
        new_parameters = parameters[l] - alpha*Mcorrected/(np.sqrt(Vcorrected) + eps)
        list_new_parameters.append(new_parameters)
    
        Mnew.append(Mnewl)
        Vnew.append(Vnewl)
                
    return list_new_parameters, Mnew, Vnew

def gradient_descent(parameters, alpha, gradients):
    L = len(parameters)
    
    list_new_parameters = []
    
    for l in range(L):
        new_parameters = parameters[l] - alpha*gradients[l]
        list_new_parameters.append(new_parameters)
        
    return list_new_parameters

#Created by Juan Manuel Espinoza Bullón - RA 228562