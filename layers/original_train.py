from layers.original_model import *
import tensorflow as tf
from layers.original_inits import get_Weight_from_file
from layers.original_utils import l2_loss

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)


def train_gcn(early_stopping=10, ephochs=200, data_str="cora", dropout=0.5, hidden_unit=16, learning_rate=0.01, weight_decay=5e-4):
    load_data_function = lambda x: load_data(data_str)
    model = GCN(load_data_function=load_data_function, hidden_unit=hidden_unit, learning_rate=learning_rate, weight_decay=weight_decay)

    cost_val = []
    # train
    for i in range(ephochs):
        # train step
        train_loss, train_acc, train_weight_decay_loss = model.one_train()
        print("train_loss", train_loss)
        print("train_acc", train_acc)
        print("train_weight_decay_loss", train_weight_decay_loss)

        weight_decay_loss = weight_decay * l2_loss(get_Weight_from_file("weights_hidden"))
        print("weight_decay_loss", weight_decay_loss)

        break
        # model.one_update()

        # val step
        # val_loss, val_acc = model.evaluate()
        #
        # print("iteration: {}, train_loss: {}, train_acc: {}, val_loss: {}, val_acc: {}".
        #       format(i, train_loss, train_acc, val_loss, val_acc))
        # cost_val.append(val_loss)
        #
        # if i > early_stopping and cost_val[-1] > np.mean(cost_val[-(early_stopping + 1): -1]):
        #     print("early stopping ! ")

    test_loss, test_acc = model.test()
    print("start test, the loss: {}, the acc: {}".format(test_loss, test_acc))

if __name__ == '__main__':
    train_gcn()