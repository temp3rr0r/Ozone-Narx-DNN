import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # These lines should be called asap, after the os import
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Use CPU only by default
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GTX 970
os.environ["PATH"] += os.pathsep + 'C:/Users/temp3rr0r/Anaconda3/Library/bin/graphviz'
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM
from keras.optimizers import SGD
import keras.backend as K
import numpy as np
import tensorflow as tf

wkdir = "storedModels"
pb_filename = "kerasToTf.pb"
# TODO: 1. Fp16 keras.
K.set_epsilon(1e-4)
K.set_floatx('float16')
print("--- Working with tensorflow.keras float precision: {}".format(K.floatx()))


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.compat.v1.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.compat.v1.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.compat.v1.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph

# TODO: 2. Create keras model.
# TODO: test FC model
# # Generate dummy data
# num_classes = 2
# x_train = np.random.random((1000, 20))
# y_train = keras.utils.to_categorical(np.random.randint(num_classes, size=(1000, 1)), num_classes=num_classes)
# x_test = np.random.random((100, 20))
# y_test = keras.utils.to_categorical(np.random.randint(num_classes, size=(100, 1)), num_classes=num_classes)
# model = Sequential()
# # Dense(64) is a fully-connected layer with 64 hidden units.
# # in the first layer, you must specify the expected input data shape:
# # here, 20-dimensional vectors.
# model.add(Dense(64, activation='relu', input_dim=20))
# model.add(Dropout(0.5))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))
# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
# model.fit(x_train, y_train, epochs=4, batch_size=128)score = model.evaluate(x_test, y_test, batch_size=128)
# score = model.evaluate(x_test, y_test, batch_size=128)
# print("Score: ", score)

# TODO: test LSTM model
data_dim = 16
timesteps = 8
num_classes = 2
# expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(LSTM(32, return_sequences=True, input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32))  # return a single vector of dimension 32
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# Generate dummy training data
x_train = np.random.random((1000, timesteps, data_dim))
y_train = np.random.random((1000, num_classes))
# Generate dummy validation data
x_val = np.random.random((100, timesteps, data_dim))
y_val = np.random.random((100, num_classes))
model.fit(x_train, y_train, batch_size=64, epochs=5, validation_data=(x_val, y_val))
score = model.evaluate(x_val, y_val, batch_size=128)
print("Score: ", score)

# TODO: 3. Store keras model as tf model.
frozen_graph = freeze_session(K.get_session(), output_names=[out.op.name for out in model.outputs])
tf.io.write_graph(frozen_graph, wkdir, pb_filename, as_text=False)

from tensorflow.python.platform import gfile
with tf.compat.v1.Session() as sess:
    # TODO: 4. Load tf model.
    # load model from pb file
    with tf.io.gfile.GFile(wkdir + "/" + pb_filename, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        g_in = tf.import_graph_def(graph_def)
    # TODO: tensorboard needed?
    # write to tensorboard (check tensorboard for each op names)
    # writer = tf.summary.FileWriter(wkdir+'/log/')
    # writer.add_graph(sess.graph)
    # writer.flush()
    # writer.close()

    # TODO: print operation names
    # print('\n===== output operation names =====\n')
    # for op in sess.graph.get_operations():
    #   print(op)

    # TODO: 5. Train tf model.
    # with tf.device("/cpu:0"):
    #     sess.run(c)
    #     sess.fit_generator(
    #         x_train,
    #         # steps_per_epoch=8000,
    #         epochs=5,
    #         validation_data=y_train,
    #         # validation_steps=2000
    #     )
    # import tensorflow.contrib.graph_editor as ge
    # # load the graphdef into memory, just as in Step 1
    # # graph = load_graph('frozen.pb')
    # graph = sess.graph
    # # create a variable for each constant, beware the naming
    # const_var_name_pairs = []
    # probable_variables = ["import/lstm_1_input", "import/dense_1/Softmax"]
    # for name in probable_variables:
    #     var_shape = graph.get_tensor_by_name('{}:0'.format(name)).get_shape()
    #     var_name = '{}_a'.format(name)
    #     var = tf.get_variable(name=var_name, shape=var_shape, dtype='float32')
    #     # var = tf.get_variable(name=var_name, shape=var_shape)
    #     const_var_name_pairs.append((name, var_name))
    # # from now we're going to work with GraphDef
    # name_to_op = dict([(n.name, n) for n in graph.as_graph_def().node])
    # # magic: now we swap the outputs of const and created variable
    # for const_name, var_name in const_var_name_pairs:
    #     const_op = name_to_op[const_name]
    #     var_reader_op = name_to_op[var_name + '/read']
    #     ge.swap_outputs(ge.sgv(const_op), ge.sgv(var_reader_op))
    # # Now we can safely create a session and copy the values
    # sess = tf.Session(graph=graph)
    # for const_name, var_name in const_var_name_pairs:
    #     ts = graph.get_tensor_by_name('{}:0'.format(const_name))
    #     var = tf.get_variable(var_name)
    #     var.load(ts.eval(sess))
    # Create a Saver object
    # Use some values for the horizontal and vertical shift
    # Create placeholders for the x and y points
    # X = tf.placeholder("float")
    # Y = tf.placeholder("float")
    # # Initialize the two parameters that need to be learned
    # h_est = tf.Variable(0.0, name='hor_estimate')
    # v_est = tf.Variable(0.0, name='ver_estimate')
    # # y_est holds the estimated values on y-axis
    # y_est = tf.square(X - h_est) + v_est
    # # Define a cost function as the squared distance between Y and y_est
    # cost = (tf.pow(Y - y_est, 2))
    # # The training operation for minimizing the cost function. The
    # # learning rate is 0.001
    # trainop = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
    # h = 1
    # v = -2
    # # Define a cost function as the squared distance between Y and y_est
    # cost = (tf.pow(Y - y_est, 2))
    # trainop = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
    # def train_graph():
    #     with tf.Session() as sess:
    #         sess.run(init)
    #         for i in range(100):
    #             for (x, y) in zip(x_train, y_train):
    #                 # Feed actual data to the train operation
    #                 sess.run(trainop, feed_dict={X: x, Y: y})
    #
    #             # Create a checkpoint in every iteration
    #             saver.save(sess, 'model_iter', global_step=i)
    #
    #         # Save the final model
    #         saver.save(sess, 'model_final')
    #         h_ = sess.run(h_est)
    #         v_ = sess.run(v_est)
    #     return h_, v_
    # saver = tf.train.Saver()
    # init = tf.global_variables_initializer()
    # # Run a session. Go through 100 iterations to minimize the cost
    # result = train_graph()
    # print("h_est = %.2f, v_est = %.2f" % result)


    # TODO: 6. Infer with tf model.
    # inference by the model (op name must comes with :0 to specify the index of its output)
    # tensor_output = sess.graph.get_tensor_by_name('import/dense_3/Softmax:0')
    # tensor_input = sess.graph.get_tensor_by_name('import/dense_1_input:0')
    tensor_output = sess.graph.get_tensor_by_name('import/dense_1/Softmax:0')
    tensor_input = sess.graph.get_tensor_by_name('import/lstm_1_input:0')
    predictions = sess.run(tensor_output, {tensor_input: x_val})
    print('\n===== output predicted results =====\n')
    print(predictions)
