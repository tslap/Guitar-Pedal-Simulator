# Below is the WaveNet described in the accompanying paper. 

class Network():
    
    class Conv_Layer():

        def __init__(self, num_kernels, input_dim_y, kernel_length, dilation, input_length, layer):
            self.num_kernels = num_kernels
            self.input_dim_y = input_dim_y
            self.kernel_length = kernel_length

            self.dilation = dilation
            self.input_length = input_length
            self.layer = layer

            self.kernel = np.random.normal(0,1/3,(num_kernels, input_dim_y, kernel_length))

            self.weights = np.random.normal(0,1/3,num_kernels)

            self.bias = np.random.normal(0,1/3,(num_kernels,input_length))
            
            self.effective_kernel_length = (self.kernel.shape[2] + 
                                                (self.dilation - 1) * (self.kernel.shape[2] -1)) 
            
        def sigmoid(self, input):
    
            if np.sum(input) > 100000 * input.shape[0]:
                return 0.99999
            
            exp = np.exp(-input.astype(float))
            return 1 / (1 + exp)

        def forward(self, input):
            
            if input.size != 2716:
                pass
            
            self.last_input = input
            
            pad_factor = ((self.effective_kernel_length - 1)) // 2 

            input = np.pad(input, ((0,0),(pad_factor,pad_factor)) )
            
            if self.dilation == 1:
                input = np.pad(input, ((0,0),(1,0)))

            self.input = input
                
            output = np.zeros((self.num_kernels, self.input_length))


            
            for nn in range(self.num_kernels):
                for tt in range(output.shape[1]):
                    for ii in range(self.kernel.shape[2]):
                        output[nn,tt] += np.sum(input[:,tt + self.dilation * ii] * self.kernel[nn,:,ii])

            z_k = self.sigmoid(output + self.bias)
                        
            self.last_z_k = z_k

            x_next = np.zeros(self.last_input.shape)
            
            for tt in range(output.shape[1]):
                x_next[:,tt] = np.dot(z_k[:,tt] , self.weights) / self.weights.size + self.last_input[:,tt]

            return z_k, x_next

        
        def backprop_sgd(self, dE_dYhat, linear_mixer , lr):

            dE_dH = np.zeros(self.kernel.shape)
            dE_dB = np.zeros(self.bias.shape)
            
            dE_dW = np.zeros(self.weights.shape)

            for nn in range(self.num_kernels):
                for tt in range(self.last_input.shape[1]):

                    dE_dH[nn,:,:] = dE_dH[nn,:,:] + (dE_dYhat[0,tt] * linear_mixer[self.layer, nn] 
                                      * self.last_z_k[nn, tt] 
                                     * (1 - self.last_z_k[nn, tt])
                                     * self.input[:, tt : tt + self.effective_kernel_length : self.dilation])


            dE_dB = dE_dYhat * linear_mixer[self.layer, nn] * self.last_z_k * (1 - self.last_z_k)

            return dE_dH , dE_dB, dE_dW

        
    def __init__(self, num_layers, num_kernels, input_dim_y, kernel_length, dilation, input_length):
        self.num_layers = num_layers
        self.num_kernels = num_kernels
        self.input_dim_y = input_dim_y
        self.kernel_length = kernel_length
        self.dilation = dilation
        self.input_length = input_length
        
        self.linear_mixer = np.random.normal(0,1/3,(num_layers,num_kernels))

        self.layers = []
      
        for ii in range(num_layers):
            self.layers.append( self.Conv_Layer(num_kernels,input_dim_y, kernel_length, dilation ** ii , input_length, ii) )

    def forward(self, y):

        zks = []
        
        out = y
        for ii in range(self.num_layers):
            zk , out = self.layers[ii].forward(out)
            zks.append(zk)
            
        zks = np.asarray(zks)
        
        y_hat = np.zeros(y.shape)
        
        for tt in range(self.input_length):            
            y_hat[:,tt] = np.sum(self.linear_mixer * zks[:,:,tt])

        return y_hat , zks
    
    def forward_sgd(self, Y):
        
        Y_hat = []
        Zks = []
        for y in Y:
            y_hat, zks = forward(y)
            Y_hat.append(y_hat)
            Zks.append(Zks)
        return
    
    def Loss(self, y, y_hat):

        sum_intensity = np.sum(y**2)
        loss = np.sum((y - y_hat) ** 2) / sum_intensity

        grad = -2 * (y - y_hat) / sum_intensity

        return loss , grad

    def backprop_lin_mixer(self,grad, zks):

        dE_dWz = np.zeros(self.linear_mixer.shape)
    
        for tt in range(self.input_length):
            dE_dWz += grad[0,tt] * zks[:,:,tt]
            
            
        return dE_dWz
  
            
    def backprop_sgd(self, dE_dYhats, linear_mixer, lr, bs):        
        
        # Want to find sum of gradients found through back prop from each data point in the batch
        # Once found want to average and update features
        
        
        # Finding Sums
        dE_dWz = np.zeros(self.linear_mixer.shape)
        
        dE_dHs = []
        dE_dBs = []
        dE_dWs = []
        for ii in range(num_layers):
            dE_dHs.append(np.zeros(self.layers[ii].kernel.shape))
            dE_dBs.append(np.zeros(self.layers[0].bias.shape))
            dE_dWs.append(np.zeros(self.layers[0].weights.shape))
            
        for jj in range(bs):
            for ii in self.layers:
                dE_dW_temp , dE_dB_temp , dE_dW_temp = ii.backprop_sgd(dE_dYhats[jj], linear_mixer, lr)
                
                dE_dWs[ii] += dE_dW_temp
                dE_dBs[ii] += dE_dB_temp
                dE_dWs[ii] += dE_dW_temp
                
            dE_dWz += self.backprop_lin_mixer(dE_dYhats[jj])

        # update each feature
        
        self.linear_mixer -= 0.01 * dE_dWz
        
        for ii in range(num_layers):
            self.layers[ii].kernel -= (dE_dHs[ii] / bs) * lr
            self.layers[ii].bias -= (dE_dBs[ii] / bs) * lr
            self.layers[ii].weights -= (dE_dWs[ii] / bs) * lr

    
    def stochastic_gradient_descent(self, data , bs, lr):
        # data - list of tuples containing input and output training data. Already shuffled when created
        # bs - batch size
        
        # SPLIT DATA INTO BATCHES
        batches = []
        for ii in range(len(data) // bs):
            batches.append(data[ii * bs:(ii+1) * bs][:])
            
        # TRAIN NEWTORK WITH EACH BATCH
        
        #establish zeros at beggining of each batch
        dE_dWz = np.zeros(self.linear_mixer.shape)
        dE_dHs = []
        dE_dBs = []
        dE_dWs = []
        for ii in range(self.num_layers):
            dE_dHs.append(np.zeros(self.layers[ii].kernel.shape))
            dE_dBs.append(np.zeros(self.layers[0].bias.shape))
            dE_dWs.append(np.zeros(self.layers[0].weights.shape))
            
        total_loss = 0
        total_loss_og = 0
        
        #Find sum of gradients
        for ii in range(len(batches)):
            print(ii)
            for jj in range(bs):
                x = batches[ii][jj][0][np.newaxis]
                y = batches[ii][jj][1][np.newaxis]
                #side note: why do I bother returning zks?
                
                if x.size == self.input_length:
                    y_hat, zks = self.forward(x)

                    loss, dE_dYhat = self.Loss(y, y_hat)
                    loss_og, _ = self.Loss(y,x)
                    total_loss += loss
                    total_loss_og += loss_og

                    # Update parameters 

                    dE_dWz += self.backprop_lin_mixer(dE_dYhat, zks)

                    for nn in range(self.num_layers):
                        dE_dH, dE_dB, dE_dW = self.layers[nn].backprop_sgd(dE_dYhat, self.linear_mixer, 1)


                        dE_dHs[nn] = dE_dHs[nn] + dE_dH
                        dE_dBs[nn] = dE_dBs[nn] + dE_dB
                        dE_dWs[nn] = dE_dWs[nn] + dE_dW

                    

            print(total_loss / bs)
            plt.figure(figsize = (12,8))
            plt.plot(x[0,:], color = '#0102FA', linewidth = 0.5, label = 'input')
            plt.plot(y[0,:], color = '#FA5319', label = 'target', linewidth = 3, alpha = 0.75)
            plt.plot( y_hat[0,:], color = 'green',alpha = 0.75, label = 'output')  
            plt.legend()
            plt.show()

            self.linear_mixer -= dE_dWz / bs * lr * 1
            for nn in range(self.num_layers):
                self.layers[nn].kernel = self.layers[nn].kernel - dE_dHs[nn] / bs * lr 
                self.layers[nn].bias = self.layers[nn].bias - dE_dBs[nn] / bs * lr 
                self.layers[nn].weights = self.layers[nn].weights - dE_dW / bs * lr


            total_loss = 0
            total_loss_og = 0
