
# coding: utf-8

# In[1]:


# Using google drive to store training and testing data.
#from google.colab import drive

# Retrieving data as objects
import pickle
# Performing fast computations
import numpy as np
# Setting uyp Neural Networks
import tensorflow as tf
# Plotting graphs
import matplotlib.pyplot as plt
# Other utilities

import cv2
# Mounting drive that contains my data.
# If you run this notebook, save the training data in a subdirectory called 'HW4-data'. Look at the next cell to better understand the structure of the directories.
# drive.mount('/content/drive')


# ## Training Set
#
#

# In[15]:


BASE_PATH = './drive/My Drive/DL Project/'
SAVED_MODELS = BASE_PATH + 'models/'


# In[2]:


# DONT BOTHER WITH THIS FOR NOW
#training_set_generator = pickle.load(open( "./drive/My Drive/DL Project/stickmap.p", "rb" ))
#training_set_discriminator1 = pickle.load(open("./drive/My Drive/DL Project/train1.p", "rb"))
#training_set_discriminator2 = pickle.load(open("./drive/My Drive/DL Project/train2.p", "rb"))
#training_set_discriminator3 = pickle.load(open("./drive/My Drive/DL Project/train3.p", "rb"))


# ## Model

# ### Generator

# In[3]:


# Add layer specific output
def generator(z):
    with tf.variable_scope("GAN/Generator", reuse=tf.AUTO_REUSE):
        # add noise to map. mostly not needed

        # Uncomment all print statements to check output shape
        # print(z)

        ######## ENCODING #######

        # Convolution 0 and Max Pooling 0
        gen_conv0 = tf.layers.conv2d(z, filters=64, kernel_size=3, strides=1, padding='SAME',
                                     activation=tf.nn.leaky_relu, kernel_initializer=tf.variance_scaling_initializer())
        # print(gen_conv0)
        gen_pool0 = tf.layers.average_pooling2d(gen_conv0, pool_size=2, strides=2, padding="VALID")
        # print(gen_pool0)

        # Convolution 1 and Max Pooling 1
        gen_conv1 = tf.layers.conv2d(gen_pool0, filters=128, kernel_size=3, strides=1, padding='SAME',
                                     activation=tf.nn.leaky_relu, kernel_initializer=tf.variance_scaling_initializer())
        # print(gen_conv1)
        gen_pool1 = tf.layers.average_pooling2d(gen_conv1, pool_size=2, strides=2, padding="VALID")
        # print(gen_pool1)

        # Convolution 2 and Max Pooling 2
        gen_conv2 = tf.layers.conv2d(gen_pool1, filters=256, kernel_size=3, strides=1, padding='SAME',
                                     activation=tf.nn.leaky_relu, kernel_initializer=tf.variance_scaling_initializer())
        # print(gen_conv2)
        gen_pool2 = tf.layers.average_pooling2d(gen_conv2, pool_size=2, strides=2, padding="VALID")
        # print(gen_pool2)

        # Convolution 3 and Max Pooling 3
        gen_conv3 = tf.layers.conv2d(gen_pool2, filters=512, kernel_size=3, strides=1, padding='SAME',
                                     activation=tf.nn.leaky_relu, kernel_initializer=tf.variance_scaling_initializer())
        # print(gen_conv3)
        gen_pool3 = tf.layers.average_pooling2d(gen_conv3, pool_size=2, strides=2, padding="VALID")
        # print(gen_pool3)

        # Convolution 4 and Max Pooling 4
        gen_conv4 = tf.layers.conv2d(gen_pool3, filters=512, kernel_size=3, strides=1, padding='SAME',
                                     activation=tf.nn.leaky_relu, kernel_initializer=tf.variance_scaling_initializer())
        # print(gen_conv4)
        gen_pool4 = tf.layers.average_pooling2d(gen_conv4, pool_size=2, strides=2, padding="VALID")
        # print(gen_pool4)

        # Convolution 5 and Max Pooling 5
        gen_conv5 = tf.layers.conv2d(gen_pool4, filters=512, kernel_size=3, strides=1, padding='SAME',
                                     activation=tf.nn.leaky_relu, kernel_initializer=tf.variance_scaling_initializer())
        # print(gen_conv5)
        gen_pool5 = tf.layers.average_pooling2d(gen_conv5, pool_size=2, strides=2, padding="VALID")
        # print(gen_pool5)

        # Convolution 6 and Max Pooling 6
        gen_conv6 = tf.layers.conv2d(gen_pool5, filters=512, kernel_size=3, strides=1, padding='SAME',
                                     activation=tf.nn.leaky_relu, kernel_initializer=tf.variance_scaling_initializer())
        # print(gen_conv6)
        gen_pool6 = tf.layers.average_pooling2d(gen_conv6, pool_size=2, strides=2, padding="VALID")
        # print(gen_pool6)

        ######## DECODING #######

        # Deconvolution 0
        gen_deconv0 = tf.layers.conv2d_transpose(gen_pool6, filters=512, kernel_size=(3, 3), strides=(
            2, 2), padding='SAME', kernel_initializer=tf.variance_scaling_initializer())
        # print(gen_deconv0)

        # Deconvolution 1
        gen_deconv1 = tf.layers.conv2d_transpose(gen_deconv0, filters=512, kernel_size=(
            3, 3), strides=(2, 2), padding='SAME', kernel_initializer=tf.variance_scaling_initializer())
        # print(gen_deconv1)

        # Deconvolution 2
        gen_deconv2 = tf.layers.conv2d_transpose(gen_deconv1, filters=512, kernel_size=(
            3, 3), strides=(2, 2), padding='SAME', kernel_initializer=tf.variance_scaling_initializer())
        # print(gen_deconv2)

        # Deconvolution 3
        gen_deconv3 = tf.layers.conv2d_transpose(gen_deconv2, filters=512, kernel_size=(
            3, 3), strides=(2, 2), padding='SAME', kernel_initializer=tf.variance_scaling_initializer())
        # print(gen_deconv3)

        # Deconvolution 4
        gen_deconv4 = tf.layers.conv2d_transpose(gen_deconv3, filters=512, kernel_size=(
            3, 3), strides=(2, 2), padding='SAME', kernel_initializer=tf.variance_scaling_initializer())
        # print(gen_deconv4)

        # Deconvolution 5
        gen_deconv5 = tf.layers.conv2d_transpose(gen_deconv4, filters=512, kernel_size=(
            3, 3), strides=(2, 2), padding='SAME', kernel_initializer=tf.variance_scaling_initializer())
        # print(gen_deconv5)

        # Deconvolution 6
        gen_deconv6 = tf.layers.conv2d_transpose(gen_deconv5, filters=512, kernel_size=(
            3, 3), strides=(2, 2), padding='SAME', kernel_initializer=tf.variance_scaling_initializer())
        # print(gen_deconv6)

        # Deconvolution 7
        gen_deconv7 = tf.layers.conv2d_transpose(gen_deconv6, filters=3, kernel_size=(3, 3), strides=(
            1, 1), padding='SAME', kernel_initializer=tf.variance_scaling_initializer())
        # print(gen_deconv7)

        return gen_deconv7


# ### Discriminator 1

# In[4]:


# Add layer specific output

def discriminator_1(z):
    with tf.variable_scope("GAN/Discriminator1", reuse=tf.AUTO_REUSE):

        # print(z)
        # Uncomment all print statements to check output shapes

        # Convolution 0 and Max Pooling 0
        dis1_conv0 = tf.layers.conv2d(z, filters=64, kernel_size=3, strides=1, padding='SAME',
                                      activation=tf.nn.relu, kernel_initializer=tf.variance_scaling_initializer())
        # print(dis1_conv0)
        dis1_pool0 = tf.layers.average_pooling2d(
            dis1_conv0, pool_size=2, strides=2, padding="VALID")
        # print(dis1_pool0)

        # Convolution 1 and Max Pooling 1
        dis1_conv1 = tf.layers.conv2d(dis1_pool0, filters=128, kernel_size=3, strides=1, padding='SAME',
                                      activation=tf.nn.relu, kernel_initializer=tf.variance_scaling_initializer())
        # print(dis1_conv1)
        dis1_pool1 = tf.layers.average_pooling2d(
            dis1_conv1, pool_size=2, strides=2, padding="VALID")
        # print(dis1_pool1)

        # Convolution 2 and Max Pooling 2
        dis1_conv2 = tf.layers.conv2d(dis1_pool1, filters=256, kernel_size=3, strides=1, padding='SAME',
                                      activation=tf.nn.relu, kernel_initializer=tf.variance_scaling_initializer())
        # print(dis1_conv2)
        dis1_pool2 = tf.layers.average_pooling2d(
            dis1_conv2, pool_size=2, strides=2, padding="VALID")
        # print(dis1_pool2)

        # Convolution 3 and Max Pooling 3
        dis1_conv3 = tf.layers.conv2d(dis1_pool2, filters=512, kernel_size=3, strides=1, padding='SAME',
                                      activation=tf.nn.relu, kernel_initializer=tf.variance_scaling_initializer())
        # print(dis1_conv3)
        dis1_pool3 = tf.layers.average_pooling2d(
            dis1_conv3, pool_size=2, strides=2, padding="VALID")
        # print(dis1_pool3)

        # Convolution 4 and Max Pooling 4
        dis1_conv4 = tf.layers.conv2d(dis1_pool3, filters=512, kernel_size=3, strides=1, padding='SAME',
                                      activation=tf.nn.relu, kernel_initializer=tf.variance_scaling_initializer())
        # print(dis1_conv4)
        dis1_pool4 = tf.layers.average_pooling2d(
            dis1_conv4, pool_size=2, strides=2, padding="VALID")
        # print(dis1_pool4)

        # Convolution 5 and Max Pooling 5
        dis1_conv5 = tf.layers.conv2d(dis1_pool4, filters=512, kernel_size=3, strides=1, padding='SAME',
                                      activation=tf.nn.relu, kernel_initializer=tf.variance_scaling_initializer())
        # print(dis1_conv5)
        dis1_pool5 = tf.layers.average_pooling2d(
            dis1_conv5, pool_size=2, strides=2, padding="VALID")
        # print(dis1_pool5)

        # Convolution 6 and Max Pooling 6
        dis1_conv6 = tf.layers.conv2d(dis1_pool5, filters=512, kernel_size=3, strides=1, padding='SAME',
                                      activation=tf.nn.relu, kernel_initializer=tf.variance_scaling_initializer())
        # print(dis1_conv6)
        dis1_pool6 = tf.layers.average_pooling2d(
            dis1_conv6, pool_size=2, strides=2, padding="VALID")
        # print(dis1_pool6)

        # Final Convolution and Flattening
        dis1_conv7 = tf.layers.conv2d(dis1_pool6, filters=1, kernel_size=3, strides=1, padding='SAME',
                                      activation=tf.nn.relu, kernel_initializer=tf.variance_scaling_initializer())
        # print(dis1_conv7)
        dis1_flat = tf.layers.flatten(dis1_conv7)
        # print(dis1_flat)
        dis1_logit = tf.layers.dense(
            dis1_flat, 1, kernel_initializer=tf.variance_scaling_initializer())
        # print(dis1_logit)

        return dis1_logit


# ### Discriminator 2

# In[5]:


# Add layer specific output

def discriminator_2(z):
    with tf.variable_scope("GAN/Discriminator2", reuse=tf.AUTO_REUSE):

        # print(z)

        # Uncomment all print statements to check output shapes

        # Convolution 0 and Max Pooling 0
        dis2_conv0 = tf.layers.conv2d(z, filters=64, kernel_size=3, strides=1, padding='SAME',
                                      activation=tf.nn.relu, kernel_initializer=tf.variance_scaling_initializer())
        # print(dis2_conv0)
        dis2_pool0 = tf.layers.average_pooling2d(
            dis2_conv0, pool_size=2, strides=2, padding="VALID")
        # print(dis2_pool0)

        # Convolution 1 and Max Pooling 1
        dis2_conv1 = tf.layers.conv2d(dis2_pool0, filters=128, kernel_size=3, strides=1, padding='SAME',
                                      activation=tf.nn.relu, kernel_initializer=tf.variance_scaling_initializer())
        # print(dis2_conv1)
        dis2_pool1 = tf.layers.average_pooling2d(
            dis2_conv1, pool_size=2, strides=2, padding="VALID")
        # print(dis2_pool1)

        # Convolution 2 and Max Pooling 2
        dis2_conv2 = tf.layers.conv2d(dis2_pool1, filters=256, kernel_size=3, strides=1, padding='SAME',
                                      activation=tf.nn.relu, kernel_initializer=tf.variance_scaling_initializer())
        # print(dis2_conv2)
        dis2_pool2 = tf.layers.average_pooling2d(
            dis2_conv2, pool_size=2, strides=2, padding="VALID")
        # print(dis2_pool2)

        # Convolution 3 and Max Pooling 3
        dis2_conv3 = tf.layers.conv2d(dis2_pool2, filters=512, kernel_size=3, strides=1, padding='SAME',
                                      activation=tf.nn.relu, kernel_initializer=tf.variance_scaling_initializer())
        # print(dis2_conv3)
        dis2_pool3 = tf.layers.average_pooling2d(
            dis2_conv3, pool_size=2, strides=2, padding="VALID")
        # print(dis2_pool3)

        # Convolution 4 and Max Pooling 4
        dis2_conv4 = tf.layers.conv2d(dis2_pool3, filters=512, kernel_size=3, strides=1, padding='SAME',
                                      activation=tf.nn.relu, kernel_initializer=tf.variance_scaling_initializer())
        # print(dis2_conv4)
        dis2_pool4 = tf.layers.average_pooling2d(
            dis2_conv4, pool_size=2, strides=2, padding="VALID")
        # print(dis2_pool4)

        # Convolution 5 and Max Pooling 5
        dis2_conv5 = tf.layers.conv2d(dis2_pool4, filters=512, kernel_size=3, strides=1, padding='SAME',
                                      activation=tf.nn.relu, kernel_initializer=tf.variance_scaling_initializer())
        # print(dis2_conv5)
        dis2_pool5 = tf.layers.average_pooling2d(
            dis2_conv5, pool_size=2, strides=2, padding="VALID")
        # print(dis2_pool5)

        # Convolution 6 and Max Pooling 6
        dis2_conv6 = tf.layers.conv2d(dis2_pool5, filters=512, kernel_size=3, strides=1, padding='SAME',
                                      activation=tf.nn.relu, kernel_initializer=tf.variance_scaling_initializer())
        # print(dis2_conv6)
        dis2_pool6 = tf.layers.average_pooling2d(
            dis2_conv6, pool_size=2, strides=2, padding="VALID")
        # print(dis2_pool6)

        # Final Convolution and Flattening
        dis2_conv7 = tf.layers.conv2d(dis2_pool6, filters=1, kernel_size=3, strides=1, padding='SAME',
                                      activation=tf.nn.relu, kernel_initializer=tf.variance_scaling_initializer())
        # print(dis2_conv7)
        dis2_flat = tf.layers.flatten(dis2_conv7)
        # print(dis2_flat)
        dis2_logit = tf.layers.dense(
            dis2_flat, 1, kernel_initializer=tf.variance_scaling_initializer())
        # print(dis2_logit)

        return dis2_logit


# ### Test Bed / Running the model

# In[6]:


# Graph setup
tf.keras.backend.set_image_data_format('channels_last')
tf.reset_default_graph()

sess = tf.Session()
# sess.run(tf.initialize_all_variables())


# In[7]:


# Placeholders
d1_data = tf.placeholder(tf.float32, shape=[None, 640, 640, 6])  # img and map
d2_data = tf.placeholder(tf.float32, shape=[None, 640, 640, 6])
posemap = tf.placeholder(tf.float32, shape=[None, 640, 640, 3])
noise = tf.placeholder(tf.float32, shape=[None, 640, 640, 1])


# In[8]:


# Pathways
discriminator_1_image_decision = discriminator_1(d1_data)
discriminator_2_image_decision = discriminator_2(d2_data)
generated_image = generator(tf.concat([noise, posemap], 3))
discriminator_1_generated_decision = discriminator_1(tf.concat([generated_image, posemap], 3))
discriminator_2_generated_decision = discriminator_2(tf.concat([generated_image, posemap], 3))
# Check shapes
print(discriminator_1_generated_decision)
print(discriminator_2_generated_decision)


# In[9]:


# Collecting variables
generator_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="GAN/Generator")
discriminator_1_variables = tf.get_collection(
    tf.GraphKeys.GLOBAL_VARIABLES, scope="GAN/Discriminator1")
discriminator_2_variables = tf.get_collection(
    tf.GraphKeys.GLOBAL_VARIABLES, scope="GAN/Discriminator2")


# In[10]:


# LOSSES

# Normal


# discriminator_1_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminator_1_image_decision,labels=tf.ones_like(discriminator_1_image_decision)) +
#                                      tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminator_1_generated_decision, labels=tf.zeros_like(discriminator_1_generated_decision)))
# discriminator_2_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminator_2_image_decision,labels=tf.ones_like(discriminator_2_image_decision)) +
#                                      tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminator_2_generated_decision, labels=tf.zeros_like(discriminator_2_generated_decision)))
# generator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminator_1_generated_decision,labels=tf.ones_like(discriminator_1_generated_decision)) +
#                                  tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminator_2_generated_decision,labels=tf.ones_like(discriminator_2_generated_decision)))


# Losses - Wasserstien (add feature matching loss)
discriminator_1_loss = tf.reduce_mean(
    discriminator_1_image_decision) - tf.reduce_mean(discriminator_1_generated_decision)
discriminator_2_loss = tf.reduce_mean(
    discriminator_2_image_decision) - tf.reduce_mean(discriminator_2_generated_decision)
generator_loss = -tf.reduce_mean(discriminator_1_generated_decision) - \
    tf.reduce_mean(discriminator_2_generated_decision)


# In[11]:


# Optimizer
discriminator_1_optimizer = (tf.train.RMSPropOptimizer(
    learning_rate=1e-4).minimize(-discriminator_1_loss, var_list=discriminator_1_variables))
discriminator_2_optimizer = (tf.train.RMSPropOptimizer(
    learning_rate=1e-4).minimize(-discriminator_2_loss, var_list=discriminator_2_variables))
generator_optimizer = (tf.train.RMSPropOptimizer(
    learning_rate=1e-4).minimize(-generator_loss, var_list=generator_variables))


# In[12]:


# Gradient clipping
clip_discriminator_1_gradient = [p.assign(tf.clip_by_value(p, -0.01, 0.01))
                                 for p in discriminator_1_variables]
clip_discriminator_2_gradient = [p.assign(tf.clip_by_value(p, -0.01, 0.01))
                                 for p in discriminator_2_variables]


# Add FM loss to Generator- wont take time


# In[13]:


# DUMMY DATA - change these data points - read file during epochs
a = np.random.rand(10, 640, 640, 6)
b = np.random.rand(10, 640, 640, 3)
c = np.random.rand(10, 640, 640, 1)


# In[ ]:


save_step = 1


# In[14]:


# Solving - put this in loop - write code to save model - ask bhushan.

saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
for epoch in range(10):
    print(epoch)

    _, d1_loss, _ = sess.run([discriminator_1_optimizer, discriminator_1_loss, clip_discriminator_1_gradient], feed_dict={
                             d1_data: a, d2_data: a, posemap: b, noise: c})
    print(d1_loss)
    _, d2_loss, _ = sess.run([discriminator_2_optimizer, discriminator_2_loss, clip_discriminator_2_gradient], feed_dict={
                             d1_data: a, d2_data: a, posemap: b, noise: c})
    print(d2_loss)
    _, gen_loss = sess.run([generator_optimizer, generator_loss], feed_dict={
                           d1_data: a, d2_data: a, posemap: b, noise: c})
    print(gen_loss)

    if epoch % save_step == 0:
        saver.save(sess, SAVED_MODELS + 'model_' + ('0000' + str(epoch))[-3:] + '.ckpt')

# Printing section
# print('=============================================')
# print(discriminator_1_image_decision)
# print(discriminator_2_image_decision)
# print(generated_image)
# print(discriminator_1_generated_decision)
# print(discriminator_2_generated_decision)


# In[ ]:


sess.close()
