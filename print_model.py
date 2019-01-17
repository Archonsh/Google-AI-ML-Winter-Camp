from keras.models import Model, load_model
from keras.utils import plot_model
import os
import sys

WORKING_FOLDER = os.curdir

########################################################################
def print_model(MODEL_NAME):
    print("Working folder %s" % WORKING_FOLDER)
    pred_model = load_model(MODEL_NAME)
    plot_model(pred_model, to_file=MODEL_NAME+'.png')
    pred_model.summary()

if __name__ == '__main__':
    MODEL_NAME = sys.argv[1]
    print_model(MODEL_NAME)
