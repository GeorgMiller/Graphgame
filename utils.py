def mi_approximation(self):

    p_xy_matrix = np.zeros((self.batch_size,self.batch_size))
    p_xy = [[] for _ in range(self.batch_size + 2)]

    for i in range(self.batch_size):
        for j in range(self.batch_size):
            p_x = self.memory[i]
            p_y = self.memory[j]
            
            for x in range(self.mini_batch_size):
        
                p_xy[i].append(np.histogram2d(p_x[x], p_y[x]))


    return p_xy


def mutual_info(self):

    mutual_info_matrix = np.zeros([self.batch_size, self.batch_size])
    #idx = np.random.randint(low=0, high=(self.actions.shape[0]-self.mini_batch_size))

    for i in range(self.batch_size):
        for j in range(self.batch_size):
            score = 0
            v1 , v2 = self.memory_actions[i][:self.mini_batch_size], self.memory_actions[j][:self.mini_batch_size]
            
            for x in range(self.mini_batch_size):
                a = v1[x]
                b = v2[x]
                score += metrics.mutual_info_score(a,b)
            
            mutual_info_matrix[i][j] = score
            mutual_info_matrix[j][i] = mutual_info_matrix[i][j]
            if i==j:
                mutual_info_matrix[i][i] = 0

    mutual_info = tf.reduce_mean(mutual_info_matrix, axis=1)
        
    return mutual_info

def cos_between(self, v1, v2):

    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)

    return np.dot(v1_u, v2_u)
    
def cosine_similarity_weights(self, w2, w3, w4, w5):
    
    noise_batch_size = tf.identity(self.batch_size,name='noise_batch_size') 

    flattened_network = tf.concat(axis=1,values=[\
            #tf.reshape(w1, [noise_batch_size, -1]),\
            tf.reshape(w2, [noise_batch_size, -1]),\
            tf.reshape(w3, [noise_batch_size, -1]),\
            tf.reshape(w4, [noise_batch_size, -1]),\
            tf.reshape(w5, [noise_batch_size, -1])])

    cosine_weights =  np.zeros([self.batch_size, self.batch_size])
    for i in range(self.batch_size):
        for j in range(self.batch_size):
            
            cosine_weights[i][j] = self.cos_between(flattened_network[i],flattened_network[j])
            cosine_weights[j][i] = cosine_weights[i][j]


    return cosine_weights


def stuff():
            #kernel_init = tf.keras.initializers.glorot_uniform(42) # VarianceScaling(scale=0.01, seed=42)
        #bias_init = tf.keras.initializers.Constant(0)

        if kernel_init = 'custom':
            uniform1 = uniform.random_uniform(-tf.math.sqrt(6/(300)),tf.math.sqrt(6/(300)))
            uniform2 = uniform.random_uniform(-tf.math.sqrt(6/(15*2)),tf.math.sqrt(6/(15*2)))
            uniform3 = uniform.random_uniform(-tf.math.sqrt(6/(100*2)),tf.math.sqrt(6/(100*2)))
            uniform4 = uniform.random_uniform(-tf.math.sqrt(6/(15*64)),tf.math.sqrt(6/(15*64)))
            uniform5 = uniform.random_uniform(-tf.math.sqrt(6/(100*64)),tf.math.sqrt(6/(100*64)))
            uniform6 = uniform.random_uniform(-tf.math.sqrt(6/(15*64)),tf.math.sqrt(6/(15*64)))
            uniform7 = uniform.random_uniform(-tf.math.sqrt(6/(100*64)),tf.math.sqrt(6/(100*64)))