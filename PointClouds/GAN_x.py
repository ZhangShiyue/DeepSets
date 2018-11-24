import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import os.path as osp
import tensorflow as tf
import tflearn
import numpy as np
import cv2
import random

from latent_3d_points.src.ae_templates import mlp_architecture_ala_iclr_18
from latent_3d_points.src.in_out import load_all_point_clouds_under_folder
from latent_3d_points.src.show3d_balls import showpoints
from latent_3d_points.external.structural_losses.tf_nndistance import nn_distance
from latent_3d_points.external.structural_losses.tf_approxmatch import approx_match, match_cost

TRAIN = True

GENE_DISC_FACTOR = 0
BATCH_SIZE = 32
Z_SIZE = 128
POINT_NUM = 2048
IMG_SIZE = (256, 256)
LR_G = 0.0001
LR_D = 0.0001
TRAIN_NUM = 3000
IMG_DIR = 'latent_3d_points/data/2D_gene'
PC_DIR = 'latent_3d_points/data/shape_net_core_uniform_samples_2048/03001627'
SAVE_DIR = 'x_gene_gene2d/03001627_emd_unfix'
AE_MODEL = 'latent_3d_points/data/single_class_ae/models.ckpt-500'


def generator(in_img, img_size, out_size):
  with tf.variable_scope("generator"):
    layer = in_img
    
    layer = tflearn.layers.conv.conv_2d(layer, 16, (3,3), strides=1, activation='relu')
    layer = tflearn.layers.conv.max_pool_2d(layer, (2,2)) #128
    layer = tflearn.layers.conv.conv_2d(layer, 32, (3,3), strides=1, activation='relu')
    layer = tflearn.layers.conv.max_pool_2d(layer, (2,2)) #64
    layer = tflearn.layers.conv.conv_2d(layer, 64, (3,3), strides=1, activation='relu')
    layer = tflearn.layers.conv.max_pool_2d(layer, (2,2)) #32
    layer = tflearn.layers.conv.conv_2d(layer, 32, (3,3), strides=1, activation='relu')
    layer = tflearn.layers.conv.max_pool_2d(layer, (2,2)) #16
    
    layer = tf.reshape(layer, [-1, img_size[0]*img_size[1]*32/16/16])
    layer = tflearn.layers.core.fully_connected(layer, 2048, activation='relu')
    layer = tflearn.layers.core.dropout(layer, keep_prob=0.8)
    layer = tflearn.layers.core.fully_connected(layer, 256, activation='relu')
    layer = tflearn.layers.core.dropout(layer, keep_prob=0.8)
    layer = tflearn.layers.core.fully_connected(layer, 128, activation='relu')
    layer = tflearn.layers.core.dropout(layer, keep_prob=0.8)
    layer = tflearn.layers.core.fully_connected(layer, out_size, activation='linear')
  
  return layer
  
def discriminator(pc, reuse = False):
  with tf.variable_scope("discriminator"):
    layer = pc
    
    layer = tflearn.layers.conv.conv_1d(layer, 64, 1, strides=1, activation='linear', weight_decay=0.001, reuse=reuse, scope='conv1')
    layer = tflearn.activations.leaky_relu(layer, alpha=0.2)
    layer = tflearn.layers.conv.conv_1d(layer, 128, 1, strides=1, activation='linear', weight_decay=0.001, reuse=reuse, scope='conv2')
    layer = tflearn.activations.leaky_relu(layer, alpha=0.2)
    layer = tflearn.layers.conv.conv_1d(layer, 256, 1, strides=1, activation='linear', weight_decay=0.001, reuse=reuse, scope='conv3')
    layer = tflearn.activations.leaky_relu(layer, alpha=0.2)
    layer = tflearn.layers.conv.conv_1d(layer, 256, 1, strides=1, activation='linear', weight_decay=0.001, reuse=reuse, scope='conv4')
    layer = tflearn.activations.leaky_relu(layer, alpha=0.2)
    layer = tflearn.layers.conv.conv_1d(layer, 512, 1, strides=1, activation='linear', weight_decay=0.001, reuse=reuse, scope='conv5')
    layer = tflearn.activations.leaky_relu(layer, alpha=0.2)
    
    num_batch = layer.get_shape()[0].value
    num_point = layer.get_shape()[1].value
    layer = tflearn.layers.conv.max_pool_1d(layer, num_point, strides=num_point, padding='valid')
    layer = tf.reshape(layer, [num_batch, -1])
    
    layer = tflearn.layers.core.fully_connected(layer, 128, activation='linear', weight_decay=0.001, reuse=reuse, scope='fc1')
    layer = tflearn.activations.leaky_relu(layer, alpha=0.2)
    layer = tflearn.layers.core.fully_connected(layer, 64, activation='linear', weight_decay=0.001, reuse=reuse, scope='fc2')
    layer = tflearn.activations.leaky_relu(layer, alpha=0.2)
    
    # output without sigmoid
    layer = tflearn.layers.core.fully_connected(layer, 1, activation='linear', weight_decay=0.001, reuse=reuse, scope='fc3')
  return layer

def GAN_model(img_size, latent_size, point_num, batch_size):
  
  img = tf.placeholder(tf.float32,shape=(batch_size,img_size[0],img_size[1],3),name='img')
  noise = tf.random_uniform(shape=(batch_size,img_size[0],img_size[1],3),minval=-0.1,maxval=0.1,name='noise')
  #image_in = tf.add(img, noise, name='img_noise')
  image_in = img
  x_gt = tf.placeholder(tf.float32,shape=(batch_size,point_num,3),name='x_gt')
  
  z_gene = generator(image_in, img_size, latent_size)
  
  encoder, decoder, enc_args, dec_args = mlp_architecture_ala_iclr_18(point_num, latent_size)

  with tf.variable_scope("single_class_ae"):      
    layer = decoder(z_gene, **dec_args)
    x_gene = tf.reshape(layer, [-1, point_num, 3])
    
  disc_real_output = discriminator(x_gt)
  disc_fake_output = discriminator(x_gene, reuse = True)
  
  return img, x_gt, z_gene, x_gene, disc_real_output, disc_fake_output

def creat_loss(real_prob, synthetic_prob, real_pc, gene_pc):
  '''
  loss_d = tf.reduce_mean(-tf.log(real_prob) - tf.log(1 - synthetic_prob))
  loss_g = tf.reduce_mean(-tf.log(synthetic_prob))
  '''
  
  loss_d_r = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real_prob), logits=real_prob))
  loss_d_f = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(synthetic_prob), logits=synthetic_prob))
  loss_d = tf.add(loss_d_r, loss_d_f)
  
  cost_p1_p2, _, cost_p2_p1, _ = nn_distance(real_pc, gene_pc)
  loss_cd = tf.reduce_mean(cost_p1_p2) + tf.reduce_mean(cost_p2_p1)
  
  match = approx_match(gene_pc, real_pc)
  loss_emd = tf.reduce_mean(match_cost(gene_pc, real_pc, match))
     
  # cd*1,emd*100
  loss_g_g = loss_emd
  loss_g_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(synthetic_prob), logits=synthetic_prob)) * 1
  loss_g = GENE_DISC_FACTOR * loss_g_d + (1-GENE_DISC_FACTOR) * loss_g_g
    
  lr_g = tf.placeholder(dtype=tf.float32, name='lr_g')
  lr_d = tf.placeholder(dtype=tf.float32, name='lr_d')
    
  generator_vars = [v for v in tf.global_variables() if v.name.startswith("generator")]
  #generator_vars = [v for v in tf.global_variables() if (v.name.startswith("generator") or v.name.startswith("single_class_ae"))]
  discriminator_vars = [v for v in tf.global_variables() if v.name.startswith("discriminator")]
  gene_opti = tf.train.RMSPropOptimizer(learning_rate=lr_g).minimize(loss_g, var_list=generator_vars)
  disc_opti = tf.train.RMSPropOptimizer(learning_rate=lr_d).minimize(loss_d, var_list=discriminator_vars)
  
  return gene_opti, disc_opti, loss_g, loss_d, lr_g, lr_d, loss_cd, loss_emd

def read_img(data_dir, img_size):
  img = []
  img_name = []
  name_list = os.listdir(data_dir)
  name_list.sort()
  img_num = len(name_list)
  if TRAIN:
    name_list = name_list[:int(img_num*0.8)-int(img_num*0.8)%BATCH_SIZE]
  else: 
    name_list = name_list[int(img_num*0.8)-int(img_num*0.8)%BATCH_SIZE:]
  for name in name_list:
    img_name.append(os.path.splitext(name)[0])
    tmp = cv2.imread(data_dir + '/' + name)
    #tmp = cv2.resize(tmp, img_size, interpolation=cv2.INTER_LINEAR)
    img.append(tmp)
  img = np.asarray(img)
  img = np.float32(img)
  #img = np.float32(img)/255
  return img, img_name


def train():
  
  img, _ = read_img(IMG_DIR, IMG_SIZE)
  img_num = img.shape[0]
  '''
  model_num = img_num / 4
  img_train = img[:int(model_num*0.8)*4-int(model_num*0.8)*4%BATCH_SIZE, : ]
  img_train_num = img_train.shape[0]
  '''
  img_train = img
  img_train_num = img_num
  print('%5d image loaded, %5d for training' % (img_num, img_train_num))
  
  all_pc_data = load_all_point_clouds_under_folder(PC_DIR, n_threads=8, file_ending='.ply', verbose=False, sort=True)
  pc_num = all_pc_data.num_examples
  pc_train = all_pc_data.point_clouds[:int(pc_num*0.8)-int(pc_num*0.8)%(BATCH_SIZE)]
  pc_train_label = all_pc_data.labels[:int(pc_num*0.8)-int(pc_num*0.8)%(BATCH_SIZE)]
  pc_train_num = len(pc_train)
  print('%5d point cloud loaded, %5d for training' % (pc_num, pc_train_num))
  
  image, x_gt, z_gene, x_gene, real_d, fake_d = GAN_model(IMG_SIZE, Z_SIZE, POINT_NUM, BATCH_SIZE)
  print('Model created.')
    
  gene_opti, disc_opti, loss_g, loss_d, lr_g, lr_d, _, _ = creat_loss(real_d, fake_d, x_gt, x_gene)  
  print('Loss created.')
  
  
  # parameter init
  generator_vars = [v for v in tf.global_variables() if v.name.startswith("generator")]
  discriminator_vars = [v for v in tf.global_variables() if v.name.startswith("discriminator")]
  gan_vars = generator_vars + discriminator_vars
  saver_gan = tf.train.Saver(var_list=gan_vars, max_to_keep=30)
  ae_vars = [v for v in tf.global_variables() if v.name.startswith("single_class_ae")]
  saver_ae = tf.train.Saver(var_list=ae_vars)
  
  #saver_all = tf.train.Saver(max_to_keep=30)
    
  # start training
  config=tf.ConfigProto()
  config.gpu_options.allow_growth=True
  config.allow_soft_placement=True
  with tf.Session(config=config) as sess: 
    sess.run(tf.global_variables_initializer())
    # print tf.trainable_variables()
    saver_ae.restore(sess, AE_MODEL)
    batch = 0
    g_lr = LR_G
    d_lr = LR_D
    print('Training start!')
    while batch < TRAIN_NUM:
      img_input = img_train[(batch*BATCH_SIZE)%img_train_num : (batch*BATCH_SIZE)%img_train_num + BATCH_SIZE, : , : , : ]
      pc_gt = pc_train[(batch*BATCH_SIZE)%pc_train_num : (batch*BATCH_SIZE)%pc_train_num + BATCH_SIZE, : ]
      #pc_gt = np.repeat(pc_gt, 4, axis=0)
      pc_name = pc_train_label[(batch*BATCH_SIZE)%pc_train_num : (batch*BATCH_SIZE)%pc_train_num + BATCH_SIZE]
      
      if (batch + 1) % 100 == 0:
        saver_gan.save(sess, SAVE_DIR + '/models/my-model-' + str(batch+1))
        #saver_all.save(sess, SAVE_DIR + '/models/my-model-' + str(batch+1))
      
      # 200 /2
      if (batch + 1) % 500 == 0:
        g_lr = g_lr / 5
        d_lr = d_lr / 5
      
        
      feed_dict = {image : img_input, x_gt : pc_gt, lr_g : g_lr, lr_d : d_lr}
      ops = [z_gene, x_gene, real_d, fake_d, loss_g, loss_d, disc_opti, gene_opti]
      latent_gene, pc_gene, disc_real_output, disc_fake_output, gene_loss, disc_loss, _, _ = sess.run(ops, feed_dict=feed_dict)
      
      if batch % 20 == 0:
        print('Batch: %5d, gene_loss: %3.5f, disc_loss: %3.5f' % (batch, gene_loss, disc_loss))
        # print('d_r: %3.5f, d_f: %3.5f' % (disc_real_output[0], disc_fake_output[0]))
        # cmd = showpoints(np.reshape(pc_gene[0],(POINT_NUM,3)))
              
      batch += 1
    
    saver_gan.save(sess, SAVE_DIR + '/models/final-model', global_step=batch)
    
    
def test():
  img, img_name = read_img(IMG_DIR, IMG_SIZE)
  img_num = img.shape[0]
  '''
  model_num = img_num / 4
  img_test = img[int(model_num*0.8)*4-int(model_num*0.8)*4%BATCH_SIZE:, : ]
  img_test_name = img_name[int(model_num*0.8)*4-int(model_num*0.8)*4%BATCH_SIZE:]
  img_test_num = img_test.shape[0]
  '''
  img_test = img
  img_test_name = img_name
  img_test_num = img_num
  print('%5d image loaded, %5d for testing' % (img_num, img_test_num))
  
  all_pc_data = load_all_point_clouds_under_folder(PC_DIR, n_threads=8, file_ending='.ply', verbose=False, sort=True)
  pc_num = all_pc_data.num_examples
  pc_test = all_pc_data.point_clouds[int(pc_num*0.8)-int(pc_num*0.8)%(BATCH_SIZE):]
  pc_test_label = all_pc_data.labels[int(pc_num*0.8)-int(pc_num*0.8)%(BATCH_SIZE):]
  pc_test_num = len(pc_test)
  print('%5d point cloud loaded, %5d for testing' % (pc_num, pc_test_num))
  
  image, x_gt, z_gene, x_gene, real_d, fake_d = GAN_model(IMG_SIZE, Z_SIZE, POINT_NUM, 1)
  _, _, _, _, _, _, loss_cd, loss_emd = creat_loss(real_d, fake_d, x_gt, x_gene)
  
  # parameter init
  generator_vars = [v for v in tf.global_variables() if v.name.startswith("generator")]
  discriminator_vars = [v for v in tf.global_variables() if v.name.startswith("discriminator")]
  gan_vars = generator_vars + discriminator_vars
  saver_gan = tf.train.Saver(var_list=gan_vars)
  ae_vars = [v for v in tf.global_variables() if v.name.startswith("single_class_ae")]
  saver_ae = tf.train.Saver(var_list=ae_vars)
  
  saver_all = tf.train.Saver(max_to_keep=30)
  '''
  f_cd = open(SAVE_DIR + '/loss_cd.txt', 'w')
  f_emd = open(SAVE_DIR + '/loss_emd.txt', 'w')
  cd_sum = 0
  emd_sum = 0
  '''
  config=tf.ConfigProto()
  config.gpu_options.allow_growth=True
  config.allow_soft_placement=True
  with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    #saver_ae.restore(sess, AE_MODEL)
    #model_file=tf.train.latest_checkpoint(SAVE_DIR + '/models')
    model_file = SAVE_DIR + '/models/my-model-2900'
    #saver_gan.restore(sess, model_file)
    saver_all.restore(sess, model_file)
    #for i in range(img_test_num):
    for i in [0,4,10,30,33,50]:
      gt = pc_test[i:i+1, : ]
      img_input = img_test[i:i+1, : ]
      ops = [x_gene, z_gene, real_d, fake_d, loss_cd, loss_emd]
      pc_gene, latent_gene , disc_real_output, disc_fake_output, cd_loss, emd_loss = sess.run(ops, feed_dict={image : img_input, x_gt : gt})
      # print('Discriminator output (real, fake): (%5d, %5d)' % (disc_real_output, disc_fake_output))
      
      cmd = showpoints(np.reshape(pc_gene[0],(POINT_NUM,3)))
      cmd = showpoints(np.reshape(gt[0],(POINT_NUM,3)))
      
      print cd_loss
      print emd_loss
      print img_test_name[i]
      print pc_test_label[i]
      '''
      f_cd.write(str(cd_loss))
      f_cd.write('\n')      
      cd_sum = cd_sum + cd_loss
      f_emd.write(str(emd_loss))
      f_emd.write('\n') 
      emd_sum = emd_sum + emd_loss
      
      f = open(SAVE_DIR + '/out/' + img_test_name[i] + '.txt', 'w')
      for i in latent_gene[0]:  
        f.write(str(i))
        f.write('\n')  
      f.close()
      
    f_cd.close()
    f_emd.close()
    print cd_sum/img_test_num
    print emd_sum/img_test_num
    '''
  
def main(argv=None):
  if TRAIN == True:
    train()
  else:
    test()

if __name__ == '__main__':
  tf.app.run()