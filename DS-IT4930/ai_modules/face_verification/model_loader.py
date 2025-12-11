from keras_facenet import FaceNet
from mtcnn import MTCNN

_facenet_model = None
_mtcnn_detector = None

def load_facenet():
    global _facenet_model
    if _facenet_model is None:
        _facenet_model = FaceNet()
    return _facenet_model

def load_mtcnn():
    global _mtcnn_detector
    if _mtcnn_detector is None:
        _mtcnn_detector = MTCNN()
    return _mtcnn_detector
