import keras
from keras import layers
import numpy as np

latent_dim=100
height=32
width=32
channels=3

################生成器
generator_input=keras.Input(shape=(latent_dim,))

x=layers.Dense(128*16*16)(generator_input)
x=layers.LeakyReLU()(x)
x=layers.Reshape((16,16,128))(x)

x=layers.Conv2D(256,5,padding='same')(x)
x=layers.LeakyReLU()(x)

x=layers.Conv2DTranspose(256,4,strides=2,padding='same')(x)
x=layers.LeakyReLU()(x)

x=layers.Conv2D(256,5,padding='same')(x)
x=layers.LeakyReLU()(x)
x=layers.Conv2D(256,5,padding='same')(x)
x=layers.LeakyReLU()(x)

x=layers.Conv2D(channels,7,activation='tanh',padding='same')(x)
generator=keras.models.Model(generator_input,x)
generator.summary()
################生成器


################判别器及其训练
discriminator_input=layers.Input(shape=(height,width,channels))
x=layers.Conv2D(128,3)(discriminator_input)
x=layers.LeakyReLU()(x)
x=layers.Conv2D(128,4,strides=2)(x)
x=layers.LeakyReLU()(x)
x=layers.Conv2D(128,4,strides=2)(x)
x=layers.LeakyReLU()(x)
x=layers.Conv2D(128,4,strides=2)(x)
x=layers.LeakyReLU()(x)
x=layers.Flatten()(x)

x=layers.Dropout(0.4)(x)#dropout层

x=layers.Dense(1,activation='sigmoid')(x)

discriminator=keras.models.Model(discriminator_input,x)
discriminator.summary()

discriminator_optimizer=keras.optimizers.RMSprop(lr=0.0008,clipvalue=1.0,decay=1e-8)#限制梯度范围和学习率衰减
discriminator.compile(optimizer=discriminator_optimizer,loss='binary_crossentropy')
################判别器及其训练


################冻结判别器训练生成器
discriminator.trainable=False
gan_input=keras.Input(shape=(latent_dim,))
gan_output=discriminator(generator(gan_input))
gan=keras.models.Model(gan_input,gan_output)
gan_optimizer=keras.optimizers.RMSprop(lr=0.0004,clipvalue=1.0,decay=1e-8)
gan.compile(optimizer=gan_optimizer,loss='binary_crossentropy')
################冻结判别器训练生成器


################输入，输出，保存图像
import os
from keras.preprocessing import image
(x_train,y_train),(_,_)=keras.datasets.cifar10.load_data()
x_train=x_train[y_train.flatten()==6]
x_train=x_train.reshape((x_train.shape[0],)+(height,width,channels)).astype('float32')/255.#批量标准化
iterations=10000
batch_size=20
save_dir='E:\\PycharmProjects\\untitled\\DCGAN'

start=0
for step in range(iterations):
    random_latent_vectors=np.random.normal(size=(batch_size,latent_dim))#随机采样
    generated_images=generator.predict(random_latent_vectors)#生成图像
    stop=start+batch_size
    real_images=x_train[start:stop]
    combined_images=np.concatenate([generated_images,real_images])#将生成图像与真实图像放在一起
    labels=np.concatenate([np.ones((batch_size,1)),np.zeros((batch_size,1))])
    labels+=0.05*np.random.random(labels.shape)#合并标签并添加随机噪音
    d_loss=discriminator.train_on_batch(combined_images,labels)#训练判别器

    random_latent_vectors=np.random.normal(size=(batch_size,latent_dim))#再次随机采样
    misleading_targets=np.zeros((batch_size,1))
    a_loss=gan.train_on_batch(random_latent_vectors,misleading_targets)#训练生成器

    start+=batch_size
    if start>len(x_train)-batch_size:
        start=0
    if step % 100 ==0:
        gan.save_weights('gan.h5')
        print('discriminator loss:',d_loss)
        print('adversarial loss:',a_loss)

        img=image.array_to_img(generated_images[0]*255.,scale=False)
        img.save(os.path.join(save_dir,'generated_face'+str(step)+'.png'))#保存一张生成图像

        img = image.array_to_img(real_images[0] * 255., scale=False)
        img.save(os.path.join(save_dir, 'real_face' + str(step) + '.png'))  # 保存一张真实图像









