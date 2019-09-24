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

def train_gcn(early_stopping=10, ephochs=200, data_str="cora", dropout=0.5, hidden_unit=16, learning_rate=0.01, weight_decay=5e-4):
    load_data_function = lambda x: load_data(data_str)
    model = GCN(load_data_function=load_data_function, hidden_unit=hidden_unit, learning_rate=learning_rate, weight_decay=weight_decay)

    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data_function(0)
    # adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = gcn_load_data("cora")
    cost_val = []
    # train
    for i in range(ephochs):
        # train step
        train_loss, train_acc, train_weight_decay_loss = model.one_train()
        print("train_loss", train_loss)
        print("acc", train_acc)
        #
        # print("weights_hidden", model.weight_hidden.dtype, model.weight_hidden[:2, :2])
        # print("weights_outputs", model.weight_outputs.dtype, model.weight_outputs[:2, :2])
        # print("hidden: ", model.hidden.dtype)
        # print("outputs: ", model.outputs[:2, :2])
        save_weight("model_outputs", model.outputs)
        outputs = get_Weight_from_file("model_outputs")
        loss = forward_cross_entrocpy_loss(model.outputs.astype(np.float32), model.y_train, model.train_mask)
        print("np loss", loss)
        # print("model y_train", model.y_train)
        # print("model train_mask", model.train_mask)

        with tf.Session() as sess:
            tf_loss = sess.run(masked_softmax_cross_entropy(outputs.astype(np.float32), model.y_train, model.train_mask))
            print("tf loss", tf_loss[0])

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

    # test_loss, test_acc = model.test()
    # print("start test, the loss: {}, the acc: {}".format(test_loss, test_acc))

if __name__ == '__main__':
    train_gcn()