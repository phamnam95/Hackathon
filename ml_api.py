from flask import Flask, request, jsonify, json
from ast import literal_eval
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
import numpy as np
import tensorflow as tf
import pickle
import model5
import os

APP = Flask(__name__)

@APP.route('/predict_from_array')


def pred_from_arr():
    """
        Predict from POST request using saved ML model
    """

    global_step = tf.Variable(0, name="global_step", trainable=False)

    with tf.name_scope("Input"):
        x = tf.placeholder(tf.float32, [None,156,156,100])
        y = tf.placeholder(tf.int32, [None,156,156,100])
        keepprob = tf.placeholder(tf.float32)
        is_training = tf.placeholder(tf.bool)
        loss, logits, conv1, upsamp1, conv_decode1, conv2, conv4, upsamp2, upsamp4, conv_decode2, conv_decode4 = model5.inference(x,y, keepprob, is_training)

    learning_rate = 0.05


    pred = tf.cast(tf.argmax(tf.nn.softmax(logits), -1), tf.int32)



    with tf.name_scope("Train"):
        grad_op = tf.train.AdamOptimizer(learning_rate)
        train_op = grad_op.minimize(loss, global_step=global_step)

    tf.summary.scalar("cost", loss)
    tf.summary.histogram('histogram_loss', loss)
    summary_op = tf.summary.merge_all()
    saver=tf.train.Saver()
    init_op=tf.global_variables_initializer()
    def parse_image(image, y_size, x_size, z_size):
       image = image[0:x_size, 0:y_size, 0:z_size]
       num_patches_per_x = image.shape[0] // 156
       num_patches_per_y = image.shape[1] // 156
       num_patches_per_z = image.shape[2] // 100
       x_size_crop = num_patches_per_x * 156
       y_size_crop = num_patches_per_y * 156
       z_size_crop = num_patches_per_z * 100
       image = image[0:x_size_crop, 0:y_size_crop, 0:z_size_crop]
       patches = image.reshape(num_patches_per_x, 156,
                            num_patches_per_y, 156, num_patches_per_z, 100)
       patches = patches.transpose(0,2,4, 1,3,5)
       patches = patches.reshape(num_patches_per_x * num_patches_per_y * num_patches_per_z,
                              156, 156, 100)
       return patches
    sess = tf.Session()
    init_op.run(session=sess)   
    ckpt = tf.train.get_checkpoint_state(os.path.dirname('~/pytorch-geo-intro/checkpoint'))
    # if that checkpoint exists, restore from checkpoint
    if ckpt and ckpt.model_checkpoint_path:
       saver.restore(sess, ckpt.model_checkpoint_path)

   
    test = pickle.load( open( "dw.p", "rb" ) )
    m=8.529666e-05
    s=0.0065450026
     
    test=(test-m)/s
    test=parse_image(test, 312, 312, 100)
    test_data=test

    test_images = np.zeros([4, 156, 156, 100])

    null_labels_test = np.zeros_like(test_images).astype(np.int32)
    output_test=None
    for i in range(4):
       test_images[i,:,:,:]=test_data[i,:,:,:]

    num_sample_generate = 300
    pred_tot = []
    var_tot = []
    prob_tot=[]

    test_output = sess.run(tf.nn.softmax(logits), feed_dict={x: test_images, y: null_labels_test, keepprob:1, is_training:True})
    if output_test is not None:
       output_test = np.concatenate((output_test, test_output), axis=0)
    else:
       output_test = test_output


    for image_batch, label_batch in zip(test_images,null_labels_test):
       image_batch = np.reshape(image_batch,[1,156,156,100])
       label_batch = np.reshape(label_batch,[1,156,156,100])
       prob_iter_tot = []
       pred_iter_tot = []
    
       for iter_step in range(num_sample_generate):
          prob_iter_step = sess.run(tf.nn.softmax(logits), feed_dict = {x: image_batch, y: label_batch, keepprob:0.7, is_training:True}) 
          prob_iter_tot.append(prob_iter_step)
          pred_iter_tot.append(np.reshape(np.argmax(prob_iter_step,axis = -1),[-1]))

    
       prob_mean = np.nanmean(prob_iter_tot,axis = 0)
       prob_variance = np.var(prob_iter_tot, axis = 0)
       pred = np.reshape(np.argmax(prob_mean,axis = -1),[-1])       
       pred = np.reshape(pred,[156,156,100])
       prob=prob_mean[0,:,:,:,1]
       chan_var=prob_variance[0,:,:,:,1]
       prob_variance=np.reshape(prob_variance,[156*156*100,2])
       variance=np.nanmean(prob_variance,axis=1)
       variance=np.reshape(variance,(156,156,100))
       pred_tot.append(prob)
       prob_tot.append(pred)
       var_tot.append(chan_var)
    pred_tot=np.array(pred_tot)

    var_tot=np.array(var_tot)
    prob=np.concatenate((np.concatenate((pred_tot[0,:,:,:],pred_tot[1,:,:,:]),axis=1),np.concatenate((pred_tot[2,:,:,:],pred_tot[3,:,:,:]),axis=1)),axis=0)
    var=np.concatenate((np.concatenate((var_tot[0,:,:,:],var_tot[1,:,:,:]),axis=1),np.concatenate((var_tot[2,:,:,:],var_tot[3,:,:,:]),axis=1)),axis=0)
    return 'Hello'
if __name__ == '__main__':
    APP.run(debug=True, host='0.0.0.0', port=5000)
