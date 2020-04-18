from keras.models import load_model
from keras import backend as K
from keras.preprocessing.image import img_to_array
from skimage.transform import resize

import matplotlib
matplotlib.use('Agg')

from matplotlib import image
import matplotlib.pyplot as plt
import numpy as np
import base64
import io
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
app = Flask(__name__)
cors = CORS(app)

sess = tf.Session()
graph = tf.get_default_graph()

# IMPORTANT: models have to be loaded AFTER SETTING THE SESSION for keras! 
# Otherwise, their weights will be unavailable in the threads after the session there has been set
K.tensorflow_backend.set_session(sess)
model = load_model('./best_model.h5')

def preprocess(img):
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img


@app.route("/", methods=['GET'])
def hello():
    return 'Hello World!'

@app.route("/", methods=['POST'])
def generateHeatmap():
    global sess
    global graph
    with graph.as_default():
        K.tensorflow_backend.set_session(sess)
        data = request.get_json()

        imgstring = data["imageData"]
        imgdata = base64.b64decode(imgstring)
        imgdata = io.BytesIO(imgdata)
        image0 = image.imread(imgdata, format='JPG')
        image_1 = preprocess(image0)

        last_conv = model.get_layer('conv5_block16_concat')
        grads = K.gradients(model.output[:, 2], last_conv.output)[0]

        pooled_grads = K.mean(grads, axis=(0, 1, 2))
        iterate = K.function([model.input], [pooled_grads, last_conv.output[0]])
        pooled_grads_value, conv_layer_output = iterate([image_1])

        for i in range(1024):
            conv_layer_output[:, :, i] *= pooled_grads_value[i]

        heatmap = np.mean(conv_layer_output, axis=-1)

        for x in range(heatmap.shape[0]):
            for y in range(heatmap.shape[1]):
                heatmap[x, y] = np.max(heatmap[x, y], 0)

        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)

        upsample = resize(heatmap, (224, 224), preserve_range=True)
        plt.axis('off')
        fig1 = plt.gcf()
        plt.imshow(image0)
        plt.imshow(upsample, alpha=0.5)
        plt.draw()

        pic_IObytes = io.BytesIO()
        fig1.savefig(pic_IObytes, format='JPG', bbox_inches='tight', pad_inches=0)
        pic_IObytes.seek(0)
        pic_hash = base64.b64encode(pic_IObytes.read()).decode()

        return jsonify({"result": pic_hash})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
