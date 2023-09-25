import tensorflow as tf
import random

AUTOTUNE = tf.data.experimental.AUTOTUNE

path = '/Users/antonioguimaraesfilho/PycharmProjects/fusion_tf24/path'
train_path = '/train/img/*.png'
trainm_path = '/train/imgm/*.png'
label_train_path = '/train/mask/*.png'
valid_path = '/valid/img/*.png'
validm_path = '/valid/imgm/*.png'
label_valid_path = '/valid/mask/*.png'
test_path = '/test/img/*.png'
testm_path = '/test/imgm/*.png'
label_test_path = '/test/mask/*.png'

"""Data Preprocessing"""


def input_solver(x, y, z):
    return {'RGB_input': x, 'DEPTH_input': y}, z


def input_solver_2ds(x, y, z):
    shape_mask = z.shape.as_list()
    return {'RGB_input': x, 'DEPTH_input': y}, {
        'output_inter': tf.image.resize(z, [shape_mask[0] // 8, shape_mask[1] // 8],
                                        method='nearest'), 'output_final': z}

def input_solver_2ds_same(x, y, z):
    return {'RGB_input': x, 'DEPTH_input': y}, {'output_inter': z, 'output_final': z}


choose = {0: input_solver, 1: input_solver_2ds, 2: input_solver_2ds_same}

# decode image RGB from tf.string to tf.image
def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.io.decode_png(img, channels=3, dtype=tf.uint8)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return img


# convert binary mask (0,255) to float (0,1)
def decode_mask(mask):
    mask = tf.io.decode_png(mask, channels=1, dtype=tf.uint8)
    mask = tf.image.convert_image_dtype(mask, tf.float32)
    return mask


def check_data(dataset):
    dataset = list(dataset.as_numpy_iterator())
    print(dataset)


# augmentation tf

def augmentation_v3(x_img, x2_img, y_mask, IMG_WIDTH, IMG_HEIGHT):
    s = random.random()
    # x_img, x2_img, y_mask = tf.image.resize(x_img, [IMG_HEIGHT, IMG_WIDTH]), \
    #                         tf.image.resize(x2_img, [IMG_HEIGHT, IMG_WIDTH]), \
    #                         tf.image.resize(y_mask, [IMG_HEIGHT, IMG_WIDTH])
    # x_img, x2_img, y_mask = tf.image.resize_with_crop_or_pad(x_img, IMG_HEIGHT, IMG_WIDTH), \
    #                         tf.image.resize_with_crop_or_pad(x2_img, IMG_HEIGHT, IMG_WIDTH), \
    #                         tf.image.resize_with_crop_or_pad(y_mask, IMG_HEIGHT, IMG_WIDTH)
    x_img, x2_img, y_mask = tf.image.random_crop(x_img, [IMG_WIDTH, IMG_HEIGHT, 3], seed=s), \
                            tf.image.random_crop(x2_img, [IMG_WIDTH, IMG_HEIGHT, 3], seed=s), \
                            tf.image.random_crop(y_mask, [IMG_WIDTH, IMG_HEIGHT, 1], seed=s)
    # x_img = tf.image.random_brightness(x_img, 0.1)
    # x_img = tf.image.random_contrast(x_img,0.1,0.2)
    x_img, x2_img, y_mask = tf.image.random_flip_left_right(x_img, seed=s), \
                            tf.image.random_flip_left_right(x2_img, seed=s), \
                            tf.image.random_flip_left_right(y_mask, seed=s)
    x_img, x2_img, y_mask = tf.image.random_flip_up_down(x_img, seed=s), \
                            tf.image.random_flip_up_down(x2_img, seed=s), \
                            tf.image.random_flip_up_down(y_mask, seed=s)
    x_img, x2_img, y_mask = tf.image.rot90(x_img, k=3), \
                            tf.image.rot90(x2_img, k=3), \
                            tf.image.rot90(y_mask, k=3)

    # x_img, x2_img, y_mask = tf.clip_by_value(x_img, clip_value_min=0.0, clip_value_max=1.0), \
    #                         tf.clip_by_value(x2_img, clip_value_min=0.0, clip_value_max=1.0), \
    #                         tf.clip_by_value(y_mask, clip_value_min=0.0, clip_value_max=1.0)

    return x_img, x2_img, y_mask


def convert_text_img_dataset_train(dataset, batch_size, AUTOTUNE, IMG_WIDTH, IMG_HEIGHT, aug, ds):
    size = len(list(dataset.as_numpy_iterator()))
    dataset = dataset.shuffle(buffer_size=size)
    dataset = dataset.map(lambda x, y, z: (decode_img(tf.io.read_file(x)), decode_img(tf.io.read_file(y)),
                                           decode_mask(tf.io.read_file(z))), num_parallel_calls=AUTOTUNE)
    # dataset = dataset.cache()
    if aug:
        dataset = dataset.map(lambda x, y, z: augmentation_v3(x, y, z, IMG_WIDTH=IMG_WIDTH, IMG_HEIGHT=IMG_HEIGHT),
                              num_parallel_calls=AUTOTUNE)

        dataset = dataset.map(choose[ds])  # ###

    else:
        dataset = dataset.map(lambda x, y, z: (tf.image.resize_with_crop_or_pad(x, IMG_HEIGHT, IMG_WIDTH),
                                               tf.image.resize_with_crop_or_pad(y, IMG_HEIGHT, IMG_WIDTH),
                                               tf.image.resize_with_crop_or_pad(z, IMG_HEIGHT, IMG_WIDTH)),
                              num_parallel_calls=AUTOTUNE)

        dataset = dataset.map(choose[ds])  # ###

    dataset = dataset.batch(batch_size=batch_size).repeat()
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset


def convert_text_img_dataset_val_test(dataset, batch_size, AUTOTUNE, IMG_WIDTH, IMG_HEIGHT, ds):
    dataset = dataset.map(
        lambda x, y, z: (decode_img(tf.io.read_file(x)), decode_img(tf.io.read_file(y)),
                         decode_mask(tf.io.read_file(z))), num_parallel_calls=AUTOTUNE)
    dataset = dataset.map(lambda x, y, z: (tf.image.resize_with_crop_or_pad(x, IMG_HEIGHT, IMG_WIDTH),
                                           tf.image.resize_with_crop_or_pad(y, IMG_HEIGHT, IMG_WIDTH),
                                           tf.image.resize_with_crop_or_pad(z, IMG_HEIGHT, IMG_WIDTH)),
                          num_parallel_calls=AUTOTUNE)

    dataset = dataset.map(choose[ds])  # ###
    # dataset = dataset.cache()
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset


def train_data(path_):  # path - dataset path / split - % train and % validation data
    dataset_img = tf.data.Dataset.list_files(path_ + train_path, shuffle=False)  # all images files
    dataset_imgm = tf.data.Dataset.list_files(path_ + trainm_path, shuffle=False)  # all 3D images files
    dataset_label = tf.data.Dataset.list_files(path_ + label_train_path, shuffle=False)  # all binary masks
    dataset_full = tf.data.Dataset.zip((dataset_img, dataset_imgm, dataset_label))  # tuple with images and labels
    # check_data(dataset_full)
    n_alldata = len(list(dataset_full.as_numpy_iterator()))  # number of images without augmentation
    dataset_train = dataset_full.shuffle(n_alldata, seed=3)  # shuffle all dataset size
    # check_data(dataset_train)
    return dataset_train

def valid_data(path_):
    dataset_img = tf.data.Dataset.list_files(path_ + valid_path, shuffle=False)  # all images files
    dataset_imgm = tf.data.Dataset.list_files(path_ + validm_path, shuffle=False)  # all 3D images files
    dataset_label = tf.data.Dataset.list_files(path_ + label_valid_path, shuffle=False)  # all binary masks
    dataset_val = tf.data.Dataset.zip((dataset_img, dataset_imgm, dataset_label))  # tuple with images and labels
    return dataset_val

def test_data(path_):
    dataset_img = tf.data.Dataset.list_files(path_ + test_path, shuffle=False)  # all images files
    dataset_imgm = tf.data.Dataset.list_files(path_ + testm_path, shuffle=False)  # all 3D images files
    dataset_label = tf.data.Dataset.list_files(path_ + label_test_path, shuffle=False)  # all binary masks
    dataset_test = tf.data.Dataset.zip((dataset_img, dataset_imgm, dataset_label))  # tuple with images and labels
    return dataset_test


# train = train_data(path)
# test = test_data(path)
# val = valid_data(path)
#
# # checkpoint
# n_train = len(list(train.as_numpy_iterator()))
# n_val = len(list(val.as_numpy_iterator()))
# n_test = len(list(test.as_numpy_iterator()))
#
#
# # def check_data(dataset):
# #     dataset = list(dataset.as_numpy_iterator())
# #     print(dataset)
#
#
# print(n_train, n_val, n_test)
# check_data(train)
# check_data(val)
# check_data(test)
#
# train = convert_text_img_dataset_train(train, 2, AUTOTUNE, 224, 224, True, 0)
# val = convert_text_img_dataset_val_test(val, 2, AUTOTUNE, 224, 224, 0)
# test = convert_text_img_dataset_val_test(test, 2, AUTOTUNE, 224, 224, 0)
#
# # image view debug
# import matplotlib.pyplot as plt
#
# for j in train.as_numpy_iterator():
#     i, p = j
#     i2, m = i['RGB_input'], i['DEPTH_input']
#     plt.imshow(i2[0,])
#     plt.imshow(m[0,])
#     plt.imshow(p[0,], cmap='gray')
#     print('chega')
