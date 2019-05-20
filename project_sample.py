from sklearn.metrics import classification_report
from keras.datasets import cifar10
import autokeras as ak
import tensorflow as tf
import keras.backend as K
import os

def get_flops(model):
    run_meta = tf.RunMetadata()
    opts = tf.profiler.ProfileOptionBuilder.float_operation()

    # We use the Keras session graph in the call to the profiler.
    flops = tf.profiler.profile(graph=K.get_session().graph,
                                run_meta=run_meta, cmd='op', options=opts)

    return flops.total_float_ops  # Prints the "flops" of the model.
	
def runAutoModel():
    # initialize the list of trianing times that we'll allow
    # Auto-Keras to train for
    TRAINING_TIMES = [
		60 * 60,		# 1 hour
    # 		60 * 60 * 2,	# 2 hours
    # 		60 * 60 * 4,	# 4 hours
    # 		60 * 60 * 8,	# 8 hours
    # 		60 * 60 * 12,	# 12 hours
    # 		60 * 60 * 24,	# 24 hours
    ]

    # load the training and testing data, then scale it into the
    # range [0, 1]
    print("[INFO] loading CIFAR-10 data...")
    ((trainX, trainY), (testX, testY)) = cifar10.load_data()
    trainX = trainX.astype("float") / 255.0
    testX = testX.astype("float") / 255.0

    # initialize the label names for the CIFAR-10 dataset
    labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    # loop over the number of seconds to allow the current Auto-Keras
    # model to train for
    for seconds in TRAINING_TIMES:
        # train our Auto-Keras model
        print("[INFO] training model for {} seconds max...".format(
            seconds))
        model = ak.ImageClassifier(verbose=True)
        model.fit(trainX, trainY, time_limit=seconds)
        model.final_fit(trainX, trainY, testX, testY, retrain=True)
        print(get_flops(model))
		print(model.summery())

        # evaluate the Auto-Keras model
        score = model.evaluate(testX, testY)
        predictions = model.predict(testX)
        report = classification_report(testY, predictions,
            target_names=labelNames)

        # write the report to disk
        p = os.path.join("{}.txt".format(seconds))
        f = open(p, "w")
        f.write(report)
        f.write("\nscore: {}".format(score))
        f.close()


if __name__ == '__main__':
    runAutoModel()
