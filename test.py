    def evaluate_network(self):

        self.batch_size = 16

        z = np.random.uniform(low = -1, high = 1, size = [self.batch_size,300])

        w1, w2, w3, w4, w5 = self.hypernetwork(z,self.batch_size)

        self.hypernetwork.load_weights('Graphgame/hypernetwork_653.000496490337.h5')
        self.graph.build()
        loss_acc = 0
        z = np.random.uniform(low = -1, high = 1, size = [self.batch_size,300])

        w1, w2, w3, w4, w5 = self.hypernetwork(z,self.batch_size)

        weights_actor = tf.concat(axis=1, values=[tf.reshape(w1[:,0:2,:],(self.batch_size,-1)), tf.reshape(w2,(self.batch_size,-1)),tf.reshape(w4,(self.batch_size,-1))])
        weights_critic = tf.concat(axis=1, values=[tf.reshape(w1[:,2:,:],(self.batch_size,-1)), tf.reshape(w3,(self.batch_size,-1)),tf.reshape(w5,(self.batch_size,-1))])


        f = np.zeros(self.batch_size)
        g = np.zeros(self.batch_size)

        for num in range(self.batch_size):
            
            self.set_weights(weights_actor, weights_critic,num)

            self.state = self.graph.start()
            done = False
            self.states, self.actions, self.next_states, self.rewards, self.dones = [],[],[],[],[] 
            
            left, right = [],[]

            while not done:
                
                state = tf.reshape(self.state, (1,-1))
                prediction = self.actor(state)[0]
                actions = [0,1,2,3]
                action = np.random.choice(actions, p=prediction.numpy())
                next_state, reward, done = self.graph.next(action)
                action_onehot = np.zeros([4])
                action_onehot[action] = 1
        
                self.states.append(self.state)
                self.actions.append(action_onehot)
                self.next_states.append(next_state)
                self.rewards.append(reward)
                self.dones.append(done)
                self.state = next_state

                if state[0][0] == 5 and 0<=state[0][1] <=5:
                    left = 'left'
                    f[num] += 1
                if state[0][1] == 5 and 0<=state[0][0] <=5:
                    right = 'right'
                    f[num] += -5

                if done:

                    self.states = np.vstack(self.states)
                    self.actions = np.vstack(self.actions)
                    self.next_states = np.vstack(self.next_states)
                    self.rewards = self.discounted_r(self.rewards)
                    self.rewards = np.vstack(self.rewards)
                    self.dones = np.vstack(self.dones)
                    
                        
                    values = self.critic(self.states)
                    next_values = self.critic(self.next_states)
                    
                    loss_critic = tf.reduce_mean(tf.math.square(values-self.rewards))*0.5

                    predictions = self.actor(self.states)

                    advantages = self.rewards - values + self.gamma*next_values*np.invert(self.dones)
                    advantages = tf.reshape(advantages, (-1))
                    pred = tf.reduce_sum(predictions*self.actions, axis=1)
                    log_pred = tf.math.log(pred + 1e-9)

                    entropy_coeff = 0.01
                    z0 = tf.reduce_sum(predictions + self.zero_fixer, axis = 1)
                    z0 = tf.stack([z0,z0,z0,z0], axis=-1)
                    p0 = predictions / z0 
                    entropy = tf.reduce_sum(p0 * (tf.math.log(p0 + self.zero_fixer)), axis=-1)
                    mean_entropy = tf.reduce_mean(entropy) 
                    entropy_loss =  mean_entropy * entropy_coeff 

                    loss_actor = - tf.reduce_mean(log_pred*advantages) + entropy_loss
                    loss_acc += loss_actor + loss_critic

                    g[num] = tf.reduce_sum(self.rewards).numpy()

        print(f, g)