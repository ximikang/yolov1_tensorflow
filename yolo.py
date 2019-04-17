import tensorflow as tf
import numpy as np
import cv2

slim = tf.contrib.slim

def leaky_relu(alpha):
    def op(inputs):
        return tf.nn.leaky_relu(inputs, alpha=alpha, name='leaky_relu')
    return op

class Yolo(object):
    def __init__(self,vervose=True):
        #flag
        self.vervose = vervose
        
        self.cell_size = 7
        self.boxes_per_cell = 2
        self.images_size = 448
        self.classes = ["aeroplane", "bicycle", "bird", "boat", "bottle",
						"bus", "car", "cat", "chair", "cow", "diningtable",
						"dog", "horse", "motorbike", "person", "pottedplant",
						"sheep", "sofa", "train","tvmonitor"]
        self.num_classes = len(self.classes)
        self.x_offset = np.transpose(np.reshape(np.array([np.arange(self.cell_size)]*self.cell_size*self.boxes_per_cell), 
                                                [self.boxes_per_cell, self.cell_size, self.cell_size]),
                                    [1, 2, 0])
        self.y_offset = np.transpose(self.x_offset, [1, 0, 2])
        self.threshold = 0.2
        self.iou_threshold = 0.4
        self.alpha = 0.1
        self.max_output_size = 10
        self.weight_path = 'YOLO_small.ckpt'
        self.images = tf.placeholder(
            tf.float32,
            [None, self.images_size, self.images_size, 3],
            name = 'images'
        )
        self.main()


    def _yolo_net(self, images, alpha):

        if self.vervose:
            print('*'*60)
            print("begin build network")
        with tf.variable_scope('yolo'):
            with slim.arg_scope(
                    [slim.conv2d, slim.fully_connected],
                    activation_fn=leaky_relu(alpha),
                    weights_regularizer=slim.l2_regularizer(0.0005),
                    weights_initializer=tf.truncated_normal_initializer(0.0, 0.01)
                ):

                net = slim.conv2d(
                            images, 64, 7, 2, padding='SAME', scope='conv_2')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_3')
                net = slim.conv2d(net, 192, 3, scope='conv_4')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_5')
                net = slim.conv2d(net, 128, 1, scope='conv_6')
                net = slim.conv2d(net, 256, 3, scope='conv_7')
                net = slim.conv2d(net, 256, 1, scope='conv_8')
                net = slim.conv2d(net, 512, 3, scope='conv_9')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_10')
                net = slim.conv2d(net, 256, 1, scope='conv_11')
                net = slim.conv2d(net, 512, 3, scope='conv_12')
                net = slim.conv2d(net, 256, 1, scope='conv_13')
                net = slim.conv2d(net, 512, 3, scope='conv_14')
                net = slim.conv2d(net, 256, 1, scope='conv_15')
                net = slim.conv2d(net, 512, 3, scope='conv_16')
                net = slim.conv2d(net, 256, 1, scope='conv_17')
                net = slim.conv2d(net, 512, 3, scope='conv_18')
                net = slim.conv2d(net, 512, 1, scope='conv_19')
                net = slim.conv2d(net, 1024, 3, scope='conv_20')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_21')
                net = slim.conv2d(net, 512, 1, scope='conv_22')
                net = slim.conv2d(net, 1024, 3, scope='conv_23')
                net = slim.conv2d(net, 512, 1, scope='conv_24')
                net = slim.conv2d(net, 1024, 3, scope='conv_25')
                net = slim.conv2d(net, 1024, 3, scope='conv_26')
                net = slim.conv2d(
                    net, 1024, 3, 2, padding='SAME', scope='conv_28')
                net = slim.conv2d(net, 1024, 3, scope='conv_29')
                net = slim.conv2d(net, 1024, 3, scope='conv_30')
                net = tf.transpose(net, [0, 3, 1, 2], name='trans_31')
                net = slim.flatten(net, scope='flat_32')
                net = slim.fully_connected(net, 512, scope='fc_33')
                net = slim.fully_connected(net, 4096, scope='fc_34')
                predicts = slim.fully_connected(
                    net, self.cell_size*self.cell_size*(self.boxes_per_cell*5+self.num_classes), activation_fn=None, scope='fc_36')
                return predicts

                
    def _detector(self, predicts):
        idx1 = self.cell_size*self.cell_size*self.num_classes
        idx2 = idx1 + self.cell_size*self.cell_size*self.boxes_per_cell
        print(predicts.shape, idx1, idx2, '!!!!!!!!!!!!!!!!!!!!!!!')
        class_probs = tf.reshape(predicts[0,:idx1], [self.cell_size, self.cell_size, self.num_classes])
        confs = tf.reshape(predicts[0,idx1:idx2], [self.cell_size, self.cell_size, self.boxes_per_cell])
        boxes = tf.reshape(predicts[0,idx2:], [self.cell_size, self.cell_size, self.boxes_per_cell, 4])

        boxes = tf.stack([(boxes[:, :, :, 0] + tf.constant(self.x_offset, dtype=tf.float32))/self.cell_size,
                          (boxes[:, :, :, 1] + tf.constant(self.y_offset, dtype=tf.float32))/self.cell_size,
                          tf.square(boxes[:, : , :, 2]),
                          tf.square(boxes[:, : , :, 3])], axis=3)
        boxes *= self.images_size
        score = tf.expand_dims(confs, -1)*tf.expand_dims(class_probs, 2)#[s,s,b,c]
        score = tf.reshape(score, [-1,self.num_classes])#[s*s*b,c]

        boxes = tf.reshape(boxes, [-1, 4])#[s*s*b,4]

        box_classes = tf.argmax(score, axis=1)#[s*s*b,1]
        box_class_socre = tf.reduce_max(score, axis=1)#[s*s*b,1]

        # filter thresold output: [s*s*b,1]
        filter_thresold = box_class_socre >= self.threshold
        score = tf.boolean_mask(box_class_socre, filter_thresold)
        boxes = tf.boolean_mask(boxes, filter_thresold)
        box_classes = tf.boolean_mask(box_classes, filter_thresold)

        #iou threshold
        #tf.image.non_max_suppression(boxes,scores,max_output_size,iou_threshold=0.5)
        #boxes [x,y,w,h] -> [xim,ymin,xmax,ymax]
        nms_boxes = tf.stack([boxes[:, 0] - 0.5 * boxes[:, 2], boxes[:, 1] - 0.5 * boxes[:, 3],
                             boxes[:, 0] + 0.5 * boxes[:, 2], boxes[:, 1] + 0.5 * boxes[:, 3]],
                             axis=1)
        nms_indices = tf.image.non_max_suppression(nms_boxes, score,
                                                   self.max_output_size,
                                                   self.iou_threshold)
        score = tf.gather(score, nms_indices)
        boxes = tf.gather(boxes, nms_indices)
        box_classes = tf.gather(box_classes, nms_indices)
        return score, boxes, box_classes


    def _load_weight(self, weights_path):
        if self.vervose:
            print('loading weight file')
        
        saver = tf.train.Saver()
        saver.restore(self.sess, weights_path)


    def detect_from_file(self, image_path, detected_image_path='detected_image.jpg', show=True):
        image = cv2.imread(image_path)
        img_h, img_w, _ = image.shape
        scores, boxes, box_classes = self._detect_from_image(image)
        print(scores, boxes, box_classes)
        predict_boxes = []
        for i in range(len(scores)):
            predict_boxes.append((self.classes[box_classes[i]],
                                  boxes[i, 0],
                                  boxes[i, 1],
                                  boxes[i, 2],
                                  boxes[i, 3],
                                  scores[i]
                                  ))
        if show:
            self.show_predict_image(image, detected_image_path, predict_boxes)


    def _detect_from_image(self, image):
        img_h, img_w, _ = image.shape
        image = cv2.resize(image, (self.images_size, self.images_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.asarray(image)

        _image = np.zeros((1, self.images_size, self.images_size, 3), dtype='float32')
        _image[0] = (image/255.0)*2.0-1.0
        scores, boxes, box_classes = self.sess.run([self.scores, self.boxes, self.box_classes],
                                                   feed_dict={self.images: _image})
        return scores, boxes, box_classes


    def show_predict_image(self, image, detected_image_path, predict_boxes):
        image = image.copy()
        #image = cv2.resize(image, (self.images_size, self.images_size))
        image_h, image_w, _ = image.shape
        w_rate = image_w/self.images_size
        h_rate = image_h/self.images_size
        for i in range(len(predict_boxes)):
            class_of_ob = predict_boxes[i][0]
            x = int(predict_boxes[i][1] * w_rate)
            y = int(predict_boxes[i][2] * h_rate)
            w = int(predict_boxes[i][3] // 2 * w_rate)
            h = int(predict_boxes[i][4] // 2 * h_rate)
            conf = predict_boxes[i][5]
            cv2.rectangle(image, (x - w, y - h), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(image, (x - w, y - h - 20), (x + w, y - h), (125, 125, 125), -1)
            cv2.putText(image, class_of_ob + ":%.2f"%conf, (x - w + 5, y - h - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.imwrite(detected_image_path, image)


    def main(self):
        image_path = "test.jpg"
        self.sess = tf.Session()
        self.scores, self.boxes, self.box_classes = self._detector(self._yolo_net(self.images, self.alpha))
        self._load_weight(self.weight_path)
        self.detect_from_file(image_path, show=True)
        


if __name__ == "__main__":
    Yolo()






