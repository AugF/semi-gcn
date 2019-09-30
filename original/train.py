from original.model import *
from original.load import *
from original.metrics import *

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

# if __name__ == '__main__':
def train_gcn(early_stopping=10, ephochs=200, data_str="cora", dropout=0.5, hidden_unit=16, learning_rate=0.01, weight_decay=5e-4):
#     early_stopping = 10; ephochs = 200; data_str = "pubmed"; dropout = 0.5; hidden_unit = 16; learning_rate = 0.01; weight_decay = 5e-4
    load_data_function = lambda x: gcn_load_data(data_str)
    model = GCN(load_data_function=load_data_function, data_str=data_str, hidden_unit=hidden_unit, learning_rate=learning_rate, weight_decay=weight_decay)

    cost_val = []
    # train
    for i in range(ephochs):
        # train step
        train_loss, train_acc = model.one_train()
        # print("model loss", train_loss)
        # print("before adam checked weight_outputs",  model.weight_outputs)
        # print("before adam checked weight_hidden",  model.weight_hidden)

        model.one_update()
        # print("model outputs", model.outputs)
        # print("model weight_outputs", model.grad_weight_outputs)
        # print("model weight_hidden", model.grad_weight_hidden)
        # print("after adam checked weight_outputs",  model.weight_outputs)
        # print("after adam checked weight_hidden",  model.weight_hidden)
        # break

        # val step
        val_loss, val_acc = model.evaluate()
        print("iteration: {}, train_loss: {}, train_acc: {}, val_loss: {}, val_acc: {}".
              format(i, train_loss, train_acc, val_loss, val_acc))
        cost_val.append(val_loss)

        if i > early_stopping and cost_val[-1] > np.mean(cost_val[-(early_stopping + 1): -1]):
            print("early stopping ! ")
            break

    # test step
    test_loss, test_acc = model.test()
    print("start test, the loss: {}, the acc: {}".format(test_loss, test_acc))
    save_weight(data_str + "_train_weight_outputs", model.weight_outputs)
    save_weight(data_str + "_train_weight_hidden", model.weight_hidden)


if __name__ == '__main__':
    train_gcn(ephochs=2, data_str="cora")
    train_gcn(ephochs=2, data_str="citeseer")
    train_gcn(ephochs=2, data_str="pubmed")
