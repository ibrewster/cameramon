import numpy
import tensorflow as tf

def read_label_file(label_file):
    """Reads a label file and returns a dictionary mapping label indices to label names."""
    with open(label_file, 'r') as f:
        labels = {}
        for line in f.readlines():
            i, label = line.strip().split(' ', 1)
            labels[int(i)] = label.strip()
        return labels
    
def input_size(interpreter):
    """Returns the input size for the model."""
    input_details = interpreter.get_input_details()
    return input_details[0]['shape'][1:3]  # Assuming the input shape is [1, height, width, channels]

def input_tensor(interpreter):
    """Returns the input tensor for the model."""
    input_details = interpreter.get_input_details()
    input_index = input_details[0]['index']
    tensor = interpreter.get_tensor(input_index)[0]
    return TensorProxy(
        tensor,
        interpreter, 
        input_index
    )

class TensorProxy:
    def __init__(self, tensor, interpreter, index):
        self._tensor = tensor
        self._interpreter = interpreter
        self._index = index
        
    def __setitem__(self, index, value):
        self._tensor[index] = value
        self._tensor = numpy.expand_dims(self._tensor, axis=0)
        self._interpreter.set_tensor(self._index, self._tensor)
    
    def __getattr__(self, attr):
        # Forward any other attribute access to the real_tensor
        return getattr(self._tensor, attr)

class DetectedObject:
    bbox = None
    id = None
    score = None
    
def get_objects(interpreter, min_confidence):
    """Extracts detected objects from the output tensor."""
    output_details = interpreter.get_output_details()
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding boxes
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class indices
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence scores

    objs = []
    for i in range(len(scores)):
        if scores[i] >= min_confidence:
            obj = DetectedObject()
            bbox = boxes[i]
            bbox = numpy.array([
                bbox[1],  # xmin
                bbox[0],  # ymin
                bbox[3],  # xmax
                bbox[2]   # ymax
            ]) * 300                
            
            obj.bbox = bbox.tolist()
            obj.id = int(classes[i])
            obj.score = scores[i]
            objs.append(obj)
    return objs

def make_interpreter(*args):
    return tf.lite.Interpreter(*args)

