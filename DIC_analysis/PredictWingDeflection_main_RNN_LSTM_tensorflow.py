import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Figure parameters =================================================
# When you insert the figure, you need to make fig height 2
plt.rcParams['font.family']     = 'sans-serif'
plt.rcParams['figure.figsize']  = 8, 6      # (w=3,h=2) multiply by 3
plt.rcParams['font.size']       = 24        # Original fornt size is 8, multipy by above number
#plt.rcParams['text.usetex']     = True
#plt.rcParams['ps.useafm'] = True
#plt.rcParams['pdf.use14corefonts'] = True
#plt.rcParams['text.latex.preamble'] = '\usepackage{sfmath}'
plt.rcParams['lines.linewidth'] = 3.   
plt.rcParams['lines.markersize'] = 8. 
plt.rcParams['legend.fontsize'] = 20        # Original fornt size is 8, multipy by above number
plt.rcParams['xtick.labelsize'] = 24        # Original fornt size is 8, multipy by above number
plt.rcParams['ytick.labelsize'] = 24        # Original fornt size is 8, multipy by above number
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in' 
plt.rcParams['figure.subplot.left']  = 0.2
#plt.rcParams['figure.subplot.right']  = 0.7
plt.rcParams['figure.subplot.bottom']  = 0.2
plt.rcParams['savefig.dpi']  = 300

tf.reset_default_graph()


class CreateInputData:
    def __init__(self, dir_savedata, name_datasave, N_data, t_trim_init):
        self.dir_savedata  = dir_savedata
        self.name_datasave = name_datasave
        self.N_data        = N_data
        self.t_trim_init   = t_trim_init

    def Import_ExpData(self, fig_flag=True):
        fps = 30.
        i_trim_init = int(self.t_trim_init*fps)  # Remove first t_trim_init [sec] from the data
        i_trim_end  = i_trim_init+self.N_data    # Remove last t_trim_end [sec] from the data
        
        # Import the data
        data_txt = np.genfromtxt(self.name_datasave, delimiter=",", skip_header=1)
        time_DIC = data_txt[:,0]
        x_disp   = data_txt[:,1]
        y_disp   = data_txt[:,2]
        dt       = time_DIC[1] - time_DIC[0]
        
        disp_TTL = -(y_disp-np.mean(y_disp[0:10]))
        vlct_TTL = np.gradient(disp_TTL,dt)
        
        # Trim data -----------------------------------------------------------
        time_rev = time_DIC[i_trim_init:i_trim_end]
        disp_rev = disp_TTL[i_trim_init:i_trim_end]
        vlct_rev = vlct_TTL[i_trim_init:i_trim_end]
        
        ExpData = {'time_rev': time_rev,
                   'disp_rev': disp_rev,
                   'vlct_rev': vlct_rev}
        
        # Plot ----------------------------------------------------------------
        if fig_flag==True:
            plt.figure('Trimmed data', figsize=(8, 10))
            plt.subplots_adjust(wspace=0.4, hspace=0.6)
            plt.subplot(211)
            plt.plot(time_rev-time_rev[0], disp_rev, 'b-')
            plt.ylabel('Disp., $u$ [pixel]')
            plt.xlim(xmin=0)
            plt.gca().tick_params(axis='x', pad=10)
            plt.gca().yaxis.set_label_coords(-0.2,0.5)
            
            plt.subplot(212)
            plt.plot(time_rev-time_rev[0], vlct_rev, 'b-')
            plt.ylabel('Vel., $\dot u$ [pixel/s]')
            plt.xlabel('Time [s]')
            plt.xlim(xmin=0)
            plt.gca().tick_params(axis='x', pad=10)
            plt.gca().yaxis.set_label_coords(-0.2,0.5)
            plt.tight_layout
                        
            plt.figure('Original Raw data')
            plt.plot(time_DIC, disp_TTL, 'k-', label = 'Raw data')
            plt.plot(time_rev, disp_rev, 'b-', label='Trimmed data for training')
            plt.legend(loc='upper right')
            plt.xlabel('Time [s]')
            plt.ylabel('Displacement [pixel]')
            
        return(ExpData)
    
    def InputDataPreparation(self, maxlen):
        ExpData = self.Import_ExpData(fig_flag=False)
        # Normalize the data (data should be 0~1) -----------------------------
        time_exp  = ExpData['time_rev']
        disp_exp  = ExpData['disp_rev']
        vlct_exp  = ExpData['vlct_rev']

        # Displacement
        d1_offset = 0.
        d1_scale  = 1. / max(abs(disp_exp))
        data0_1   = disp_exp * d1_scale
        # velocity
        d2_offset = 0.
        d2_scale  = 1. / max(abs(vlct_exp))
        data0_2   = vlct_exp * d2_scale
        
        scaleFactor = {'d1_offset': d1_offset, 'd1_scale': d1_scale,
                       'd2_offset': d2_offset, 'd2_scale': d2_scale}
        
        print('Total number of data points = {0}'.format(self.N_data) )
        data   = []
        data2  = []
        target = []
        target2 = []
    
        for i in range(0, self.N_data - maxlen):
            data.append(data0_1[i: i + maxlen])
            data2.append(data0_2[i: i + maxlen])
            target.append(data0_1[i + maxlen])
            target2.append(data0_2[i + maxlen])
        print('Duration of training data = {0} [s]'.format(time_exp[-1]-time_exp[0]))
        X = np.zeros((len(data), maxlen, 2))
        X[:,:,0] = np.array(data).reshape(len(data), maxlen)
        X[:,:,1] = np.array(data2).reshape(len(data2), maxlen)
        
        Y = np.zeros((len(target), 2))
        Y[:,0] = np.array(target)
        Y[:,1] = np.array(target2)
    
        plt.figure('Normalized Input data for training', figsize=(8, 10))
        plt.subplots_adjust(wspace=0.4, hspace=0.6)
        plt.title('Data is normalized')
        plt.subplot(211)
        plt.plot(time_exp, data0_1, 'k-', label='Whole training data')
        plt.plot(time_exp[0:maxlen], X[0,:,0], 'b-',label='First maxlen data')
        plt.plot(time_exp[maxlen:], Y[:,0], 'g--',label='Last maxlen data')
        plt.legend(bbox_to_anchor=(0, 1.02), loc='lower left', borderaxespad=0)
        plt.ylabel('Data 1')
        plt.xlim(xmin=min(time_exp))
        plt.ylim(-1, 1)
        plt.gca().tick_params(axis='x', pad=10)
        plt.gca().yaxis.set_label_coords(-0.1,0.5)
        
        plt.subplot(212)
        plt.plot(time_exp, data0_2, 'k-')
        plt.plot(time_exp[0:maxlen], X[0,:,1], 'b-')
        plt.ylabel('Data 2')
        plt.xlabel('Time [sec]')
        plt.xlim(xmin=min(time_exp))
        plt.ylim(-1, 1)
        plt.gca().tick_params(axis='x', pad=10)
        plt.gca().yaxis.set_label_coords(-0.1,0.5)
        plt.tight_layout
        return(ExpData, X, Y, scaleFactor)

class RNN_LSTM(object):
    def __init__(self, n_in, n_hidden, n_out, maxlen, dir_modelsave, patience=0, verbose=0):
        # Set random seeds
        self.seed_train = 1234
        np.random.seed(self.seed_train)
        tf.set_random_seed(self.seed_train)
        # Structure of RNN
        self.n_in     = n_in
        self.n_hidden = n_hidden
        self.n_out    = n_out
        self.maxlen   = maxlen
        # Save model
        self.dir_modelsave = dir_modelsave
        # For early stopping
        self._step = 0
        self._loss = float('inf')
        self.patience = patience
        self.verbose  = verbose
        # Design model
        self.model = self.design_model()
    
    def design_model(self):
        x = tf.placeholder(tf.float32, shape=[None, self.maxlen, self.n_in])
        t = tf.placeholder(tf.float32, shape=[None, self.n_out])
        n_batch = tf.placeholder(tf.int32, shape=[])
    
        cell = tf.contrib.rnn.LSTMCell(self.n_hidden, forget_bias=1.0)
        initial_state = cell.zero_state(n_batch, tf.float32)
    
        state = initial_state
        outputs = []  # store output from hidden layer at previous time step
        with tf.variable_scope('LSTM'):
            for tt in range(self.maxlen):
                if tt > 0:
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(x[:, tt, :], state)
                outputs.append(cell_output)
    
        output = outputs[-1]
    
        V = self.weight_variable([self.n_hidden, self.n_out])
        c = self.bias_variable([self.n_out])
        y = tf.matmul(output, V) + c  # Linear activation
                
        model = {'x': x, 't': t, 'n_batch': n_batch, 'y': y}
        
        return(model)
    
    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)
    
    def bias_variable(self, shape):
        initial = tf.zeros(shape, dtype=tf.float32)
        return tf.Variable(initial)
    
    
    def loss(self, y, t):
        mse = tf.reduce_mean(tf.square(y - t))
        return mse
    
    
    def training(self, loss, learning_rate):
        optimizer = \
            tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999)
    
        train_step = optimizer.minimize(loss)
        return train_step
    
    
    def fit(self, X_data, Y_data, scaleFactor,
            nb_epoch, batch_size, learning_rate):
        # Separate the input data into train and test data --------------------
        N_train      = int(np.size(X_data,0) * 0.9)
        N_validation = np.size(X_data, 0) - N_train
    
        X_train, X_validation, Y_train, Y_validation = \
            train_test_split(X_data, Y_data, test_size=N_validation)
        
        # Preparation for the training ----------------------------------------
        x, t, n_batch, y = self.model['x'], self.model['t'], self.model['n_batch'], self.model['y']
        
        loss = self.loss(y, t)
        train_step = self.training(loss, learning_rate)
    
        _history = { 'val_loss': [] }
        # Training process ----------------------------------------------------    
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session() as sess:
#        sess = tf.Session()
            # Run the initializer
            sess.run(init)
            
            # For evaluation
#            self._n_batch = n_batch
            self._x = x
            self._y = y
            self._sess = sess
            
            n_batches = N_train // batch_size
        
            for epoch in range(nb_epoch):
                X_, Y_ = shuffle(X_train, Y_train)
        
                for i in range(n_batches):
                    start = i * batch_size
                    end = start + batch_size
        
                    sess.run(train_step, feed_dict={
                        x: X_[start:end],
                        t: Y_[start:end],
                        n_batch: batch_size
                    })
        
                # Evaluate by using validation data
                val_loss = loss.eval(session=sess, feed_dict={
                    x: X_validation,
                    t: Y_validation,
                    n_batch: N_validation
                })
        
                _history['val_loss'].append(val_loss)
                print('epoch:', epoch,
                      ' validation loss:', val_loss)
        
                # Check Early Stopping
                if self.early_stopping_validate(val_loss):
                    break
            # Save result -----------------------------------------------------
            saver.save(sess, self.dir_modelsave+'/model.ckpt')
            filename = 'TrainingRecipe.txt'
            self.SaveTrainingRecipe(filename, N_data, N_train,
                                    nb_epoch, batch_size, learning_rate)
            np.save(self.dir_modelsave+'/X_data.npy', X_data)
            np.save(self.dir_modelsave+'/Y_data.npy', Y_data)
            np.save(self.dir_modelsave+'/scaleFactor.npy', scaleFactor)
            
            # Plot ------------------------------------------------------------
            plt.figure('val_loss')
            plt.plot(_history['val_loss'], 'b-')
            plt.xlabel('Epoch')
            plt.ylabel('Validation loss')
            plt.xlim(xmin=0)
            plt.savefig(self.dir_modelsave+'/val_loss.png')
        return _history
    
    def early_stopping_validate(self, loss):
        if self._loss < loss:
            self._step += 1
            if self._step > self.patience:
                if self.verbose:
                    print('early stopping')
                return True
        else:
            self._step = 0
            self._loss = loss

        return False
    
    def SaveTrainingRecipe(self, filename, N_data, N_train,
                           nb_epoch, batch_size, learning_rate):
        with open(self.dir_modelsave+'/'+filename, 'w') as f:
            f.write('seed_train = {0}\n'.format(self.seed_train) )
            f.write('n_in       = {0}\n'.format(self.n_in) )
            f.write('n_hidden   = {0}\n'.format(self.n_hidden) )
            f.write('n_out      = {0}\n'.format(self.n_out) )
            f.write('maxlen     = {0}\n'.format(self.maxlen) )
            f.write('nb_epoch   = {0}\n'.format(nb_epoch) )
            f.write('batch_size = {0}\n'.format(batch_size) )
            f.write('learning_rate = {0}\n'.format(learning_rate) )
            f.write('patience   = {0}\n'.format(self.patience) )
            f.write('verbose    = {0}\n'.format(self.verbose) )
            f.write('# Training data information\n')
            f.write('N_data     = {0} #Total number of data points (train + validation)\n'.format(N_data) )
            f.write('N_train    = {0} #Total number of trainig data points\n'.format(N_train) )
        return()
            
            
    
    def Prediction(self, ExpData, scaleFactor, initial_data, L_predict):
        # Import original experimental data
        time_exp  = ExpData['time_rev']
        disp_exp  = ExpData['disp_rev']
        vlct_exp  = ExpData['vlct_rev']
        dt = time_exp[1] - time_exp[0]
        with tf.Session() as sess:
            # Import saved data
            x, n_batch, y = self.model['x'], self.model['n_batch'], self.model['y']
            saver = tf.train.Saver()
            saver.restore(sess, self.dir_modelsave+'/model.ckpt')
            
            
            Z = initial_data  # Extract only first part of the data
        
            time_original  = [time_exp[i] for i in range(self.maxlen)]
            
            data_predict = []
            time_predict = []
            
            for i in range(L_predict - self.maxlen + 1):
                # Predict based on the last data set
                z_ = Z[-1:]
                y_ = y.eval(session=sess, feed_dict={
                    x: Z[-1:],
                    n_batch: 1
                })
                # Create a new time-series data based on the prediction
                sequence_ = np.concatenate(
                    (z_.reshape(self.maxlen, self.n_in)[1:], y_), axis=0) \
                    .reshape(1, self.maxlen, self.n_in)
                Z = np.append(Z, sequence_, axis=0)
                data_predict.append(y_.reshape(-1))
                time_predict.append( (self.maxlen+i)*dt+time_exp[0] )
        
            data_predict = np.array(data_predict)
            time_predict = np.array(time_predict)
            # Rescale data ----------------------------------------------------
            d1_offset, d1_scale = scaleFactor['d1_offset'], scaleFactor['d1_scale']
            d2_offset, d2_scale = scaleFactor['d2_offset'], scaleFactor['d2_scale']
            data_predict[:,0] = (data_predict[:,0]/d1_scale) + d1_offset
            data_predict[:,1] = (data_predict[:,1]/d2_scale) + d2_offset
            
            initial_data1 = (initial_data[0,:,0]/d1_scale) + d1_offset
            initial_data2 = (initial_data[0,:,1]/d2_scale) + d2_offset
            
            # Plot results ----------------------------------------------------
            alpha = 0.5
            
            plt.figure('Comparison', figsize=(8, 10))
            plt.subplots_adjust(top=0.8, bottom=0.1)
            plt.subplots_adjust(wspace=0.4, hspace=0.6)
            plt.subplot(211)
            plt.plot(time_exp, disp_exp, linestyle='solid', color='blue', label='Original experiment data')
            plt.plot(time_original, initial_data1, linestyle='dashed', color='black', label='Input data for prediction')
            plt.plot(time_predict, data_predict[:,0], color='red', alpha=alpha, label='Predicted data')
            plt.legend(bbox_to_anchor=(0, 1.05), loc='lower left', borderaxespad=0)
            plt.ylabel('Tip defl. [pixel]')
#            plt.xlim(xmin=0)
#            plt.xlim(xmax=28)
            plt.gca().yaxis.set_label_coords(-0.2,0.5)
            
            plt.subplot(212)
            plt.plot(time_exp, vlct_exp, linestyle='solid', color='blue')
            plt.plot(time_original, initial_data2, linestyle='dashed', color='black')
            plt.plot(time_predict, data_predict[:,1], color='red', alpha=alpha)
            plt.ylabel('Vel., $\dot u$ [pixel/s]')
            plt.xlabel('Time [s]')
#            plt.xlim(xmin=0)
#            plt.xlim(xmax=28)
            plt.gca().yaxis.set_label_coords(-0.2,0.5)
            plt.tight_layout

            
            plt.show()
        return(data_predict, time_predict)
    
    


    
if __name__ == '__main__':
    # =============================================================================
    # Parameters
    # =============================================================================
    # Structure of RNN
    n_in     = 2
    n_hidden = 80
    n_out    = 2
    # Model
    nb_epoch   = 500
    batch_size = 10
    maxlen     = 900 #700
    # Training
    learning_rate = 0.001
    # Early stopping
    patience = 10
    verbose  = 1
    # Save model
    dir_modelsave = 'model' #
    # Prediction
    L_predict = 2000 # Number of data points for prediction
    # =============================================================================
    # Create trainig data
    # =============================================================================
    dir_savedata  = 'TrainingData'
    name_datasave = 'AnalyzedData_cruise.txt'
    N_data        = 1500 #1500
    t_trim_init   = 37.
    inputdata = CreateInputData(dir_savedata, name_datasave, N_data, t_trim_init)
    # Import and Plot experimental data
    inputdata.Import_ExpData()
    
    # Create training data from experimental data
    ExpData, X_data, Y_data, scaleFactor = inputdata.InputDataPreparation(maxlen)

    # =============================================================================
    # Train RNN and Predict
    # =============================================================================
    rnn = RNN_LSTM(n_in, n_hidden, n_out, maxlen, dir_modelsave, patience, verbose)

    # Trainig
    _history = rnn.fit(X_data, Y_data, scaleFactor,
                       nb_epoch, batch_size, learning_rate)
    
    # Prediction
    data_predict, time_predict = rnn.Prediction(ExpData, scaleFactor,
                                                X_data[:1], L_predict)
    
    

