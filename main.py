import tensorflow as tf

print(tf.__version__)

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import PIL
from tensorflow.keras import layers
import time

import IPython
from IPython import display

from model import *


# 使用 epoch 数生成单张图片
def display_image(epoch_no):
    return PIL.Image.open('./output/image_at_epoch_{:04d}.png'.format(epoch_no))


if __name__ == '__main__':
    # Prepare and load datasets: MNIST
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

    # Preprocessing
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5  # 将图片标准化到 [-1, 1] 区间内

    BUFFER_SIZE = 60000
    BATCH_SIZE = 256

    # 批量化和打乱数据
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    generator = make_generator_model()  # Make G
    discriminator = make_discriminator_model()

    noise = tf.random.normal([1, 100])

    # 该方法返回计算交叉熵损失的辅助函数
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    # Save checkpoints
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)

    # Save outputs
    output_dir = './output'
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Train phase
    EPOCHS = 50
    noise_dim = 100
    num_examples_to_generate = 16

    # 我们将重复使用该种子（因此在动画 GIF 中更容易可视化进度）
    seed = tf.random.normal([num_examples_to_generate, noise_dim])

    # See model.py
    train(train_dataset,
          epochs=EPOCHS,
          generator=generator,
          discriminator=discriminator,
          generator_optimizer=generator_optimizer,
          discriminator_optimizer=discriminator_optimizer,
          cross_entropy=cross_entropy,
          noise_dim=noise_dim,
          batch_size=BATCH_SIZE,
          checkpoint=checkpoint,
          checkpoint_prefix=checkpoint_prefix,
          seed=seed)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    display_image(EPOCHS)

    anim_file = os.path.join(output_dir, 'dcgan.gif')

    with imageio.get_writer(anim_file, mode='I') as writer:
        filenames = glob.glob(os.path.join(output_dir, 'image*.png'))
        filenames = sorted(filenames)
        last = -1
        for i, filename in enumerate(filenames):
            frame = 2 * (i**0.5)
            if round(frame) > round(last):
                last = frame
            else:
                continue
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)

    if IPython.version_info > (6, 2, 0, ''):
        display.Image(filename=anim_file)
