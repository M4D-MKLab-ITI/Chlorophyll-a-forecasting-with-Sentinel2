# -*- coding: utf-8 -*-

import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
tf.__version__
import time
from IPython import display
import shutil
from osgeo import gdal
import datetime
import matplotlib.image
import random
from PIL import Image
import pathlib
import ast

# INPUT variables
# The user should only declare the folder containing the chla tifs 
# as well the folder that will be the main folder for the analysis

folder_fromR="Add full path here"
main_folder="Put here full path of basic folder"
chla_folder="Put here full path of original images"
checkpoint_dir = 'Put here full path of original images of training_checkpoints'#'./training_checkpoints'
txt_results_dir = "Put here full path of txt outcomes"
log_dir="Put here full path of logs/"#"logs/"

locations_json=os.listdir([folder_fromR+"/locations"][0])


locations=[]
for i in range(len(locations_json)):
    name=locations_json[i]
    name_splitted=name.split(".")[0]
    locations.append(name_splitted)



for var_var in range(len(locations_json)):
    print(var_var)
    
    main_folder=[folder_fromR+"/"+locations[var_var]][0]
    chla_folder=[main_folder+"/"+"chla"][0]    
    tif_files=glob.glob(main_folder+"/"+'*.tif')

    os.mkdir(chla_folder)
    
    # fetch all files
    for var_var_j in range(len(tif_files)):
        # construct full file path
        element=tif_files[var_var_j]
        source = element
        element_splitted=element.split("\\")
        destination = [element_splitted[0] + "/chla/" + element_splitted[1]][0]
        # copy only files
        if os.path.isfile(source):
            shutil.copy(source, destination)
            print('copied', element_splitted[1])



#removing 5 elements to make the TEST dataset
num_to_select = 1         # set the number to select here.

#removing 0 more elements to make the VALIDation dataset
num_to_select2 = 0


def from_tif_to_test(num_to_select, num_to_select2, main_folder, chla_folder):

    our_files_yeah=[main_folder+"/te_tr_val"][0]
    our_files_yeah_outputs=[main_folder+"/created_from_gans"][0]
    
    #create a folder that contains the chla_files ordered 1-> the oldest, n-> the newest
    name_of_the_folder="rename_data"
     
    #rename the elements according to the dates   
    def reshape_time(name_of_the_folder):
        rename_name=[main_folder+"/"+name_of_the_folder]
        os.mkdir(rename_name[0])
        #first we copy to a new folder
        source_folder = chla_folder + "/"
        destination_folder = rename_name[0] +"/"
        
        # fetch all files
        for file_name in os.listdir(source_folder):
            # construct full file path
            source = source_folder + file_name
            destination = destination_folder + file_name
            # copy only files
            if os.path.isfile(source):
                shutil.copy(source, destination)
                print('copied', file_name)
            
        #make list and change names for the first time
        initial_names=os.listdir(destination_folder)
        dates_list=[]
        for i in range(len(initial_names)):
            x = initial_names[i][11:19]
            str_di=[str(x[0:4])+"/"+x[4:6] +"/"+x[6:8]][0]
            #di = datetime.strptime(str_di, "%Y/%m/%d")
            dates_list.append(str_di)    
            
        
        sorted_dates=sorted(dates_list)
        
        for i in range(len(sorted_dates)):
            x = sorted_dates[i]
            x = [str(x[0:4])+str(x[5:7]) +str(x[8:10])][0]
            new_name = [rename_name[0]+"/" + str(i) +".tif"][0] 
            for j in range(len(initial_names)):
                y=initial_names[j][11:19]
                old_name=[rename_name[0]+"/" + initial_names[j]][0] 
                if x==y:
                    #print(j)
                   # if not os.path.exists(new_name):
                       os.rename(old_name, new_name)

        
    reshape_time(name_of_the_folder) #creates folder rename_data 
    

    #reading the data and deviding it into 2 consecutive categories: A and B
    # A: 1 to n-1 ,  B: 2 to n
    
    def get_matrices(name_of_the_folder):
        rename_name=[main_folder+"/"+name_of_the_folder][0]
        chla_elements=os.listdir(rename_name)
        chla_elements_an_folders=[]
        for i in range(len(chla_elements)):
            chla_elements_an_folders.append([rename_name +"/" +str(i)+".tif"])
    
        chla_list=[]
        for i in range(len(chla_elements)):
            filepath=chla_elements_an_folders[i]
            raster = gdal.Open(filepath[0])
            band = raster.GetRasterBand(1) #we are considering only band 1
            hmm=band.ReadAsArray()
         
            chla_list.append(hmm)
        return chla_list
    
    
    to_start=get_matrices(name_of_the_folder)
    
    dimentions=to_start[0].shape #heig , wid
    
    #change to_start array: replace - to 0 
    #len(to_start) 
    for i in to_start:
        i[i < 0] = 0
       
    #create file for all data outputs
    all_name=[main_folder+"/all_data"]
    os.mkdir(all_name[0])
    
    
    #creating a data base new -  (x,y,layers)  because of :  https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imsave.html
    # The image data. The shape can be one of MxN (luminance), MxNx3 (RGB) or MxNx4 (RGBA).
    new=np.stack(to_start, axis=2)
    
    matrix_mean_values=[]
    for i in range(len(to_start)):
        matrix_mean_values.append(np.mean(new[:,:,i]))
        
    matrix_sum_values=[]
    for i in range(len(to_start)):
        matrix_sum_values.append(np.sum(new[:,:,i]))    
        
    #creating 3 bands reduces the size of the data from n to n-3
    
    #also: to solve the following problem: Floating point image RGB values must be in the 0..1 range.
    #we need to rescale between 0 and 1
    
    def noisy(image):
      row,col,ch= image.shape
      mean = 1
      var = 0.2
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,(row,col,ch))
      gauss = gauss.reshape(row,col,ch)
      noisy = image + gauss
      return noisy
    
    
    new2=new
    

    max_value_of_all=np.amax(new2)
    min_value_of_all=np.amin(new2)
    
    new_resc=(new2-min_value_of_all)/(max_value_of_all-min_value_of_all)  #normalization -> [0,1]
    
    for i in range(len(to_start)-3): 
        #print(i)      
        matplotlib.image.imsave(all_name[0]+"/"+"time_s"+ str(i) +".jpg", new_resc[:,:,(i+0):(i+3)])
    
    #create file for data A
    A_name=[main_folder+"/A_data"]
    os.mkdir(A_name[0])
    
    #create file for data B
    B_name=[main_folder+"/B_data"]
    os.mkdir(B_name[0])
    
    
    source_folder = all_name[0] + "/"
    destination_folder = A_name[0] +"/"
    
    
    # fetch all files
    for file_name in os.listdir(source_folder):
        # construct full file path
        source = source_folder + file_name
        destination = destination_folder + file_name
        # copy only files
        if os.path.isfile(source):
            shutil.copy(source, destination)
            print('copied', file_name)
    
    
    source_folder = all_name[0] + "/"
    destination_folder = B_name[0] +"/"
    
    # fetch all files
    for file_name in os.listdir(source_folder):
        # construct full file path
        source = source_folder + file_name
        destination = destination_folder + file_name
        # copy only files
        if os.path.isfile(source):
            shutil.copy(source, destination)
            print('copied', file_name)
    
    
    os.remove(A_name[0]+"/time_s"+ str(len(to_start)-4) + ".jpg")
    os.remove(B_name[0]+"/time_s"+ str(0) +".jpg")
    
    AB_name=[main_folder+"/AB_data"]
    os.mkdir(AB_name[0])
    
    
    for i in range(len(os.listdir(A_name[0]))):
        im1 = Image.open(A_name[0]+"/time_s"+ str(i) + ".jpg")
        im2 = Image.open(B_name[0]+"/time_s"+ str(i+1) + ".jpg")
        
        plt.figure()
        plt.imshow(im1)

        def get_concat_h(im1, im2):
            dst = Image.new('RGB', (im1.width + im2.width, im1.height))
            dst.paste(im1, (0, 0))
            dst.paste(im2, (im1.width, 0))
            return dst
        
        
        #transform back to jpg
        get_concat_h(im1, im2).save(AB_name[0]+"/"+str(i)+"_"+str(i+1)+'.jpg')
    
    
    te_tr_val_name=[main_folder+"/te_tr_val"][0]
    os.mkdir(te_tr_val_name)
    
    test_dir=[te_tr_val_name+"/"+"test"][0]
    train_dir=[te_tr_val_name+"/"+"train"][0]
    val_dir=[te_tr_val_name+"/"+"val"][0]
    os.mkdir(test_dir)
    os.mkdir(train_dir)
    os.mkdir(val_dir)
        
    group_of_items = range(len(os.listdir(AB_name[0])))  # a sequence or set will work here.
    list_of_random_items = random.sample(group_of_items, num_to_select)
    
    ignore=list_of_random_items #list of indices to be ignored
    new_group = [ind for ind in range(len(os.listdir(AB_name[0]))) if ind not in ignore]
    
    list_of_random_items2 = random.sample(new_group, num_to_select2)
    
    ignore2=list_of_random_items2
    end = [ind for ind in new_group if ind not in ignore2]
    
    main_folder_splitted=main_folder.split("/")
    name_of_area = main_folder_splitted[len(main_folder_splitted)-1]
    
    
    for i in range(len(os.listdir(AB_name[0]))):
        source=[AB_name[0]+"/"+str(i)+"_"+str(i+1)+'.jpg'][0]
        
        if i in end: #train
            destination=[train_dir+"/"+ name_of_area  +"_" + str(i)+"_"+str(i+1)+'.jpg'][0]
            shutil.copy(source, destination)
        
        if i in list_of_random_items: #test
            destination=[test_dir+"/"+ name_of_area  +"_" + str(i)+"_"+str(i+1)+'.jpg'][0]
            shutil.copy(source, destination)
        
        if i in list_of_random_items2: #validation
            destination=[val_dir+"/"+ name_of_area  +"_" + str(i)+"_"+str(i+1)+'.jpg'][0]
            shutil.copy(source, destination)
            
    #delete temporary folders
    shutil.rmtree(A_name[0])
    shutil.rmtree(B_name[0])
    shutil.rmtree(AB_name[0])
    shutil.rmtree(all_name[0])
    
    shutil.rmtree([main_folder+"/"+name_of_the_folder][0])
    
    return(max_value_of_all, matrix_mean_values, matrix_sum_values, min_value_of_all, dimentions)


save_data_from_initial=[]

for var_var in range(len(locations_json)):
    print(var_var)
    main_folder=[folder_fromR+"/"+locations[var_var]][0]
    chla_folder=[main_folder+"/"+"chla"][0]
    function_result=from_tif_to_test(num_to_select, num_to_select2, main_folder, chla_folder)
    save_data_from_initial.append([function_result[0],function_result[1],function_result[2],function_result[3], locations[var_var], function_result[4]])

    
# Merging merge the folders into one in the ...fromR location
# and the folder will go everywhere, instead of our_files_yeah

with open(txt_results_dir+'/save_data_from_initial.txt', 'w') as f:  #write to txt to use it later
    f.write(str(save_data_from_initial))


our_files_yeah=[folder_fromR+"/te_tr_val"][0]
our_files_yeah_outputs=[folder_fromR+"/created_from_gans"][0]

our_files_yeah_test=[our_files_yeah+"/"+"test"][0]
our_files_yeah_train=[our_files_yeah+"/"+"train"][0]
our_files_yeah_val=[our_files_yeah+"/"+"val"][0]

os.mkdir(our_files_yeah)
os.mkdir(our_files_yeah_outputs)

os.mkdir(our_files_yeah_test)
os.mkdir(our_files_yeah_train)
os.mkdir(our_files_yeah_val)




for var_var in range(len(locations_json)):
    print(var_var)
    main_folder=[folder_fromR+"/"+locations[var_var]][0]
    
    main_folder_test=[main_folder+"/"+ "te_tr_val" + "/"+  "test"][0]
    main_folder_train=[main_folder+"/"+ "te_tr_val" + "/"+"train"][0]
    main_folder_val=[main_folder+"/"+ "te_tr_val" +"/"+"val"][0]
    
    source_folder = main_folder_test + "/"
    destination_folder = our_files_yeah_test +"/"

    # fetch all files
    for file_name in os.listdir(source_folder):
        # construct full file path
        source = source_folder + file_name
        destination = destination_folder + file_name
        # copy only files
        if os.path.isfile(source):
            shutil.copy(source, destination)
            print('copied', file_name)
            
    source_folder = main_folder_train + "/"
    destination_folder = our_files_yeah_train +"/"

    # fetch all files
    for file_name in os.listdir(source_folder):
        # construct full file path
        source = source_folder + file_name
        destination = destination_folder + file_name
        # copy only files
        if os.path.isfile(source):
            shutil.copy(source, destination)
            print('copied', file_name)
            
    source_folder = main_folder_val + "/"
    destination_folder = our_files_yeah_val +"/"

    # fetch all files
    for file_name in os.listdir(source_folder):
        # construct full file path
        source = source_folder + file_name
        destination = destination_folder + file_name
        # copy only files
        if os.path.isfile(source):
            shutil.copy(source, destination)
            print('copied', file_name)
    
    main_folder_chla=[main_folder+"/"+ "chla"][0]
    main_folder_te_tr_val=[main_folder+"/"+ "te_tr_val"][0]
        
    
    shutil.rmtree(main_folder_chla)
    shutil.rmtree(main_folder_te_tr_val)    
    
    

path_to_zip  = pathlib.Path(our_files_yeah)
PATH = path_to_zip


first_elemet=os.listdir([our_files_yeah+"/train"][0])[0]

sample_image = tf.io.read_file(str([our_files_yeah+"/train/"+first_elemet][0]))
sample_image = tf.io.decode_jpeg(sample_image)
dimentions=sample_image.shape

plt.figure()
plt.imshow(sample_image)

#image_file= str([our_files_yeah+"/train/"+first_elemet][0])


#Separating the images
def load(image_file):
  # Read and decode an image file to a uint8 tensor
  image = tf.io.read_file(image_file)
  image = tf.io.decode_jpeg(image)

  # Split each image tensor into two tensors:
  # - one with a real building facade image
  # - one with an architecture label image 
  w = tf.shape(image)[1]
  w = w // 2
  
  # Convert both images to float32 tensors

  input_image = image[:, :w, :]
  real_image = image[:, w:, :]
 
  input_image = tf.cast(input_image, tf.float32)
  real_image = tf.cast(real_image, tf.float32)  
 

  return input_image, real_image

inp, re = load(str([our_files_yeah+"/train/"+first_elemet][0]))

# Casting to int for matplotlib to display the images
plt.figure()
plt.imshow(inp / 255.0)
plt.figure()
plt.imshow(re / 255.0)


# Resize to 256 x 256
def resize(input_image, real_image, height, width):
  input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  return input_image, real_image

inp, re = resize(inp, re, 256,256)

plt.figure()
plt.imshow(inp / 255.0)
plt.figure()
plt.imshow(re / 255.0)


# The facade training set consist of a "buffer_size" number of images
BUFFER_SIZE = len(os.listdir([our_files_yeah+"/train"][0]))
# The batch size of 1 produced better results for the U-Net in the original pix2pix experiment
BATCH_SIZE = 1
# Each image is 256x256 in size
IMG_WIDTH = 256
IMG_HEIGHT = 256



# The values ​​in dimensions in resize in jitter must be greater than width, height
def random_crop(input_image, real_image):
  stacked_image = tf.stack([input_image, real_image], axis=0)
  cropped_image = tf.image.random_crop(
      stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])
  return cropped_image[0], cropped_image[1]

# Normalizing the images to [-1, 1]
def normalize(input_image, real_image):
  input_image = (input_image / 127.5) - 1
  real_image = (real_image / 127.5) - 1
  return input_image, real_image


@tf.function()
def random_jitter(input_image, real_image):
  # Resizing to 286x286
  input_image, real_image = resize(input_image, real_image, int(286),int(286))

  # Random cropping back to 256x256
  input_image, real_image = random_crop(input_image, real_image)

  if tf.random.uniform(()) > 0.5:
    # Random mirroring
    input_image = tf.image.flip_left_right(input_image)
    real_image = tf.image.flip_left_right(real_image)

  return input_image, real_image


plt.figure(figsize=(6, 6))
for i in range(4):
  rj_inp, rj_re = random_jitter(inp, re)
  plt.subplot(2, 2, i + 1)
  plt.imshow(rj_inp / 255.0)
  plt.axis('off')
plt.show()




def load_image_train(image_file):
  input_image, real_image = load(image_file)
  input_image, real_image = resize(input_image, real_image, 256, 256)
  input_image, real_image = random_jitter(input_image, real_image)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image

def load_image_test(image_file):
  input_image, real_image = load(image_file)
  input_image, real_image = resize(input_image, real_image,
                                    IMG_HEIGHT, IMG_WIDTH)
  input_image, real_image = normalize(input_image, real_image)
  return input_image, real_image

train_dataset = tf.data.Dataset.list_files(str(PATH / 'train/*.jpg'))

train_dataset = train_dataset.map(load_image_train,
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)

try:
  test_dataset = tf.data.Dataset.list_files(str(PATH / 'test/*.jpg'))
except tf.errors.InvalidArgumentError:
  test_dataset = tf.data.Dataset.list_files(str(PATH / 'val/*.jpg'))


test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.batch(BATCH_SIZE)


OUTPUT_CHANNELS = 3
def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result

down_model = downsample(3, 4)
down_result = down_model(tf.expand_dims(inp, 0))
print (down_result.shape)


def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result
  
up_model = upsample(3, 4)
up_result = up_model(down_result)
print (up_result.shape)


def Generator():
  inputs = tf.keras.layers.Input(shape=[256, 256, 3]) #ΕΔΩ, πρέπει να αλλάξουμε το μέγεθος

  down_stack = [
    downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
    downsample(128, 4),  # (batch_size, 64, 64, 128)
    downsample(256, 4),  # (batch_size, 32, 32, 256)
    downsample(512, 4),  # (batch_size, 16, 16, 512)
    downsample(512, 4),  # (batch_size, 8, 8, 512)
    downsample(512, 4),  # (batch_size, 4, 4, 512)
    downsample(512, 4),  # (batch_size, 2, 2, 512)
    downsample(512, 4),  # (batch_size, 1, 1, 512)
  ]

  up_stack = [
    upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
    upsample(512, 4),  # (batch_size, 16, 16, 1024)
    upsample(256, 4),  # (batch_size, 32, 32, 512)
    upsample(128, 4),  # (batch_size, 64, 64, 256)
    upsample(64, 4),  # (batch_size, 128, 128, 128)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh')  # (batch_size, 256, 256, 3)

  x = inputs

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

generator = Generator()
tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64) #random plot appeared


gen_output = generator(inp[tf.newaxis, ...], training=False)
plt.imshow(gen_output[0, ...])



#GENERATOR LOSS
LAMBDA = 100
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(disc_generated_output, gen_output, target):
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  # Mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

  total_gen_loss = gan_loss + (LAMBDA * l1_loss)

  return total_gen_loss, gan_loss, l1_loss



#DISCRIMINATOR
def Discriminator():
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
  tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')

  x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)

  down1 = downsample(64, 4, False)(x)  # (batch_size, 128, 128, 64)
  down2 = downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
  down3 = downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
  conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

  last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

  return tf.keras.Model(inputs=[inp, tar], outputs=last)


discriminator = Discriminator()
tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=64)

disc_out = discriminator([inp[tf.newaxis, ...], gen_output], training=False)
plt.imshow(disc_out[0, ..., -1], vmin=-20, vmax=20, cmap='RdBu_r')
plt.colorbar()

#DISCRIMINATOR LOSS

def discriminator_loss(disc_real_output, disc_generated_output):
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss



#optimizers and checkpoints
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


#GENERATE IMAGES AND TEST THE FUNCTION

def generate_images(model, test_input, tar):
  prediction = model(test_input, training=True)
  plt.figure(figsize=(15, 15))

  display_list = [test_input[0], tar[0], prediction[0]]
  title = ['Input Image', 'Ground Truth', 'Predicted Image']

  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    # Getting the pixel values in the [0, 1] range to plot.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.show()
  
  return(display_list)
  
for example_input, example_target in test_dataset.take(1):
  generate_images(generator, example_input, example_target)



#TRAINING

summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


@tf.function
def train_step(input_image, target, step):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_output = generator(input_image, training=True)

    disc_real_output = discriminator([input_image, target], training=True)
    disc_generated_output = discriminator([input_image, gen_output], training=True)

    gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

  generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))

  with summary_writer.as_default():
    tf.summary.scalar('gen_total_loss', gen_total_loss, step=step//1000)
    tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step//1000)
    tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step//1000)
    tf.summary.scalar('disc_loss', disc_loss, step=step//1000)
    
def fit(train_ds, test_ds, steps):
  example_input, example_target = next(iter(test_ds.take(1)))
  start = time.time()

  for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
    if (step) % 1000 == 0:
      display.clear_output(wait=True)

      if step != 0:
        print(f'Time taken for 1000 steps: {time.time()-start:.2f} sec\n')

      start = time.time()

      generate_images(generator, example_input, example_target)
      print(f"Step: {step//1000}k")

    train_step(input_image, target, step)

    # Training step
    if (step+1) % 10 == 0:
      print('.', end='', flush=True)


    # Save (checkpoint) the model every 5k steps (το αλλάζω σε 1k)
    if (step + 1) % 3000 == 0: #used to be 1000
      checkpoint.save(file_prefix=checkpoint_prefix)
      
      
      

      
#load_ext tensorboard   #must run before fit!!!
##reload_ext tensorboard
#tensorboard --logdir D:/PathoSAT/spyder_sat/logs/ #{log_dir}

fit(train_dataset, test_dataset, steps=400000)


#http://127.0.0.1:8080/                               
#python -m tensorboard.main --logdir=D:\PathoSAT\spyder_sat\logs\ --port 8080 --host 127.0.0.1 




#tensorboard dev upload --logdir {log_dir}

# Caution: This command does not terminate. 
# It's designed to continuously upload the results of long-running experiments. 
# Once your data is uploaded you need to stop it using the "interrupt execution" 
# option in your notebook tool.

# display.IFrame(
#     src="https://tensorboard.dev/experiment/lZ0C6FONROaUMfjYkVyJqw",
#     width="100%",
#     height="1000px")

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))





PATH = pathlib.Path(txt_results_dir+'/te_tr_val')

#Test dataset
test_dataset = tf.data.Dataset.list_files(str(PATH / 'test/*.jpg'))
names_of_inputs=list(test_dataset.as_numpy_iterator())


elem_num=0
names_of_inputs[elem_num]




#here we clear the folder "val"
shutil.rmtree([str(PATH / 'val')][0])
os.mkdir([str(PATH / 'val')][0])

this_specific=str(names_of_inputs[elem_num])
element_splitted=this_specific.split("'")
element_splitted=element_splitted[1].split("\\")
file_name=element_splitted[len(element_splitted)-1]

source_folder = str(PATH / 'test/')
destination_folder = str(PATH / 'val/')
# construct full file path
source = source_folder + "\\" + file_name
destination = destination_folder + "\\" + file_name
# copy only files
if os.path.isfile(source):
    shutil.copy(source, destination)
    print('copied', file_name)
    


#prediction
val_dataset = tf.data.Dataset.list_files(str(PATH / 'val/*.jpg'))
val_dataset = val_dataset.map(load_image_test)
val_dataset = val_dataset.batch(BATCH_SIZE)
what_is_that_list=[]
for inp, tar in val_dataset:
  new_value=generate_images(generator, inp, tar) #here we produce the new psuedo-image ,,,, the generative part !!!

#3/27/23 trying to get save_data_from_initial from txt

with open(txt_results_dir+'/all_locations/save_data_from_initial.txt', 'r') as file:
   content = file.read()
# Convert the string to a list of lists using ast.literal_eval()
parsed_list = ast.literal_eval(content)
save_data_from_initial = parsed_list
   

name_of_area=file_name.split("_")[0]
found_name=-5
for find_name in range(len(save_data_from_initial)):
    if file_name.split("_")[2] == save_data_from_initial[find_name][4]:
        found_name=find_name        
        get_data_from=save_data_from_initial[found_name]   
    
max_overal_value_before_norm=get_data_from[0]
min_overal_value_before_norm=get_data_from[3]
dimentions=get_data_from[5]



what_is_that=new_value

results_directly_from_gans_1=np.array(what_is_that[0])    #previous (input image)      # t1,t2,t3
results_directly_from_gans_2=np.array(what_is_that[1])    #next   (ground truth)       # t2,t3,t4 
results_directly_from_gans_3=np.array(what_is_that[2])    #forecast  (predicted)       # t2,t3,t4


#revert back to original units
#Undo the normalization
gans_created_previous_t_test=(what_is_that[0]+1)*127.5  
#Undo the resize
wttf = tf.image.resize(gans_created_previous_t_test, [int(dimentions[0]), int(dimentions[1])],
                              method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
#undo the first normalization
inp_new=(wttf/255) * ((max_overal_value_before_norm-min_overal_value_before_norm))+min_overal_value_before_norm #
gans_created_previous_t=np.array(inp_new)
gans_created_previous_t_save=gans_created_previous_t   # called 'input'


gans_created_previous_t_test=(what_is_that[1]+1)*127.5
test_that_A=(what_is_that[1]+1)*127.5

wttf = tf.image.resize(gans_created_previous_t_test, [int(dimentions[0]), int(dimentions[1])], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

inp_new=(wttf/255) * ((max_overal_value_before_norm-min_overal_value_before_norm))+min_overal_value_before_norm #
gans_created_next_t=np.array(inp_new)
gans_created_next_t_save=gans_created_next_t  # called 'ground truth'


gans_created_previous_t_test=(what_is_that[2]+1)*127.5
test_that_B=(what_is_that[2]+1)*127.5

wttf = tf.image.resize(gans_created_previous_t_test, [int(dimentions[0]), int(dimentions[1])],
                              method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

inp_new=(wttf/255) * ((max_overal_value_before_norm-min_overal_value_before_norm))+min_overal_value_before_norm #
gans_created_forecasted=np.array(inp_new)
gans_created_forecasted_save=gans_created_forecasted  # called 'predicted image'

only_t4_next=gans_created_next_t[:,:,2]
only_t4_forec=gans_created_forecasted[:,:,2]

only_t3_next=gans_created_next_t[:,:,1]
only_t3_forec=gans_created_forecasted[:,:,1]

only_t2_next=gans_created_next_t[:,:,0]
only_t2_forec=gans_created_forecasted[:,:,0]

 
# Calculation of metrics
import math
import pandas as pd
from scipy.stats import spearmanr
from scipy.stats import linregress
MSE = np.square(np.subtract(only_t4_next,only_t4_forec)).mean() 
#print('MSE: {}'.format(MSE))

RMSE = math.sqrt(MSE)
print('RMSE: {}'.format(RMSE))

pearson = np.corrcoef(only_t4_next.flatten(), only_t4_forec.flatten())[0,1]
print('Pearson Correlation:', pearson)

corr, p_value = spearmanr(only_t4_next.flatten(), only_t4_forec.flatten())
print("Spearman Correlation: {}".format(corr))


#T2 plot
x = only_t2_next.flatten()
y = only_t2_forec.flatten()

# Calculate linear regression parameters
slope, intercept, r_value, p_value, std_err = linregress(x, y)

# Create regression line
regression_line = slope * x + intercept

plt.title('T2 with Regression Line')
plt.scatter(x, y, s=1, color='black')
plt.xlabel('T2 Ground Truth')
plt.ylabel('T2 Predicted')

# Plot the regression line
plt.plot(x, regression_line,color='red')
plt.show()


#T3 plot
x = only_t3_next.flatten()
y = only_t3_forec.flatten()

# Calculate linear regression parameters
slope, intercept, r_value, p_value, std_err = linregress(x, y)

# Create regression line
regression_line = slope * x + intercept

plt.title('T3 with Regression Line')
plt.scatter(x, y, s=1, color='black')
plt.xlabel('T3 Ground Truth')
plt.ylabel('T3 Predicted')
# Plot the regression line
plt.plot(x, regression_line,color='red')
plt.show()


#T4 plot
x = only_t4_next.flatten()
y = only_t4_forec.flatten()

# Calculate linear regression parameters
slope, intercept, r_value, p_value, std_err = linregress(x, y)

# Create regression line
regression_line = slope * x + intercept

plt.plot(x, regression_line,'-r', label='Regression Line')
plt.scatter(x, y, s=1, color='black')
plt.title('T4 with Regression Line')
plt.xlabel('T4 Ground Truth')
plt.ylabel('T4 Predicted')
# Plot the regression line

plt.show()






#Create a table with all the info included
correlation_values = {'Pearson': pearson,
                      'Spearman': corr,
                      'RMSE': RMSE}

# Create a DataFrame from the dictionary
table = pd.DataFrame(data=correlation_values, index=[0])

# Display the DataFrame
print(table)



#To produce images (jpeg)

max_a=np.amax(gans_created_previous_t)
max_b=np.amax(gans_created_next_t)
max_c=np.amax(gans_created_forecasted)

min_a=np.amin(gans_created_previous_t)
min_b=np.amin(gans_created_next_t)
min_c=np.amin(gans_created_forecasted)


range_of_results= max(max_a,max_b,max_c)-min(min_a,min_b,min_c)


gans_created_previous_t_for_tifs=(gans_created_previous_t-min(min_a,min_b,min_c))/range_of_results
gans_created_next_t_for_tifs=(gans_created_next_t-min(min_a,min_b,min_c))/range_of_results
gans_created_forecasted_for_tifs=(gans_created_forecasted-min(min_a,min_b,min_c))/range_of_results



matplotlib.image.imsave(our_files_yeah_outputs+"/"+first_elemet.split("_")[0]+"_previous"+".jpg", gans_created_previous_t_for_tifs)
matplotlib.image.imsave(our_files_yeah_outputs+"/"+first_elemet.split("_")[0]+"_next"+".jpg", gans_created_next_t_for_tifs)
matplotlib.image.imsave(our_files_yeah_outputs+"/"+first_elemet.split("_")[0]+"_gans_forecasted"+".jpg", gans_created_forecasted_for_tifs)
use_that_to_measure=max(max_a,max_b,max_c)/255
print("use_that_to_measure:")
print(use_that_to_measure)



#save output (predicted) jpg to /output folder
filter_name = str(names_of_inputs[elem_num]).split("pre_")[1].split(".jpg")[0]
imgs = os.listdir(our_files_yeah_outputs)
for img in imgs:
    if img.endswith("forecasted.jpg"):
        pred_img_in_created_from_gans = img

shutil.copy(our_files_yeah_outputs+"/"+pred_img_in_created_from_gans,folder_fromR+"/predicted_img/"+pred_img_in_created_from_gans)
os.rename(folder_fromR+"/predicted_img/"+pred_img_in_created_from_gans,folder_fromR+"/predicted_img/"+filter_name+".jpg")

