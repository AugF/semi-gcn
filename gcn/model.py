

class GCN(object):
    """basic gcn model"""
    def __init__(self, input_size, hidden_size, outpus_size, std=1e-4):
        """init"""
        self.params = {}
        self.params["w1"] = 0


    def loss(self, X, y=None, res=0.0):
        """compute loss"""
        loss = 0
        grads = 0
        return loss, grads

    def train(self, X, y, X_val, y_val, learning_rate=1e-3,
              learning_rate_decay=0.95, reg=5e-6, num_iters=100,
              batch_size=200, verbose=False):
        """train"""
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        # use sgd to optimize the parameters in self.mode
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            X_batch = None
            y_batch = None

            # TODO: Crate a random minibatch of training data and lables
            pass

            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            # TODO: use the gradients in the grads dictonary to update the parameters of the network

            if verbose and it % 100 == 0:
                print("iteration %d / %d: loss %f" (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

            return {
                "loss_history" : loss_history,
                "train_acc_history": train_acc_history,
                "val_acc_history": val_acc_history
            }

    def predict(self, X):
        """predict"""
        y_pred = None

        return y_pred