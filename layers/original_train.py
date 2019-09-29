from layers.original_model import *
from layers.original_load import *
from layers.original_metrics import *
import tensorflow as tf

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)


def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss), loss, mask

if __name__ == '__main__':
# def train_gcn(early_stopping=10, ephochs=200, data_str="cora", dropout=0.5, hidden_unit=16, learning_rate=0.01, weight_decay=5e-4):
    early_stopping = 10; ephochs = 200; data_str = "cora"; dropout = 0.5; hidden_unit = 16; learning_rate = 0.01; weight_decay = 5e-4
    load_data_function = lambda x: load_data(data_str)
    model = GCN(load_data_function=load_data_function, hidden_unit=hidden_unit, learning_rate=learning_rate, weight_decay=weight_decay)

    cost_val = []
    # train
    for i in range(ephochs):
        # train step
        train_loss, train_acc = model.one_train()
        # print("model loss", train_loss)
        # print("model outputs", model.outputs)
        # print("model weight_hidden", model.weight_hidden)

        adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data_function(0)

        # model.one_update()

        # the real result
        y = y_train.copy()
        mask = train_mask.copy()
        f = lambda x: forward_cross_entrocpy_loss(x, y, mask)

        outputs = model.outputs.copy()
        grad_real = numerical_grad(f, outputs)
        print("grad_real", grad_real)
        break
        # val step
        # val_loss, val_acc = model.evaluate()
        # print("iteration: {}, train_loss: {}, train_acc: {}, val_loss: {}, val_acc: {}".
        #       format(i, train_loss, train_acc, val_loss, val_acc))
        # cost_val.append(val_loss)
        #
        # if i > early_stopping and cost_val[-1] > np.mean(cost_val[-(early_stopping + 1): -1]):
        #     print("early stopping ! ")

    # test step
    # test_loss, test_acc = model.test()
    # print("start test, the loss: {}, the acc: {}".format(test_loss, test_acc))

