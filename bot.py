import vk_api
from datetime import datetime
from vk_api.bot_longpoll import VkBotLongPoll, VkBotEventType
from token_container import get_token as gt
from vk_api.keyboard import VkKeyboard, VkKeyboardColor
import random
from tensorflow import keras
from PIL import Image
import requests
import numpy as np
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import functools
import tensorflow as tf
import PIL.Image
import os
from vk_api import VkUpload




def create_keyboard1():
    keyboard = VkKeyboard(one_time=False)
    keyboard.add_button('Оценить', color=VkKeyboardColor.POSITIVE)
    keyboard.add_line()
    keyboard.add_button('Стиль', color=VkKeyboardColor.POSITIVE)
    keyboard.add_line()
    keyboard.add_button('Помощь', color=VkKeyboardColor.PRIMARY)
    return keyboard.get_keyboard()

def create_keyboard2():
    keyboard = VkKeyboard(one_time=True)
    keyboard.add_button('назад', color=VkKeyboardColor.NEGATIVE)
    return keyboard.get_keyboard()

def image_from_url(url, size):
    p = requests.get(url)
    out = open("IMG_FROM_URL.jpg", "wb")
    out.write(p.content)
    out.close()
    original_image = Image.open("IMG_FROM_URL.jpg")
    resized_image = original_image.resize(size)
    resized_image.save("IMG_FROM_URL.jpg")


def send_message(vk_session, id_type, id, message=None, attachment=[], keyboard=None):
    vk_session.method('messages.send',{id_type: id, 'message': message, 'random_id': random.randint(-2147483648, +2147483648), "attachment": ','.join(attachment), 'keyboard': keyboard})

def img_judge(url, model):
    image_from_url(url, (128,128))
    arr = np.asarray(Image.open('IMG_FROM_URL.jpg').convert('L'))
    arr = np.around(np.array([arr]) / 255, 5)
    return model.predict(arr)

def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)
    plt.imshow(image)
    if title:
        plt.title(title)

def load_img(path_to_img):
    max_dim = 500
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img


def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


# model_2 = keras.models.load_model('MODEL_MEM_LIKES_84.h5')
model_ayy = keras.models.load_model('Model_ayyimao_4.h5')
model_mdk = keras.models.load_model('Model_mdk_5.h5')
my_token = gt()
vk_session = vk_api.VkApi(token= my_token)
upload = VkUpload(vk_session)
session_api = vk_session.get_api()
longpoll = VkBotLongPoll(vk_session, 197572047)


hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

statuses_judge = dict()


while True:
    for event in longpoll.listen():
        if event.type == VkBotEventType.MESSAGE_NEW:

            if event.obj['message']['peer_id'] not in statuses_judge:
                 statuses_judge.update({event.obj['message']['peer_id']:[False, False]})

            response = event.obj['message']['text'].lower()

            print('Get message time:\n {}'.format(datetime.now()))
            print('Message_text:\n {}'.format(event.obj['message']['text']))

            if response == 'начать' and not statuses_judge[event.obj['message']['peer_id']][0] and not statuses_judge[event.obj['message']['peer_id']][1]:
                send_message(vk_session, 'user_id', event.obj['message']['peer_id'], 'Привет! ')
                send_message(vk_session, 'user_id', event.obj['message']['peer_id'], 'Если хочешь оценить мем, нажми на кнопку или напиши "Оценить" \n Желаешь изменить стиль фотки? Тогда нажми кнопку или напиши мне "Стиль"', keyboard=create_keyboard1())

            if statuses_judge[event.obj['message']['peer_id']][0]:
                if event.obj['message']['attachments'] and len(event.obj['message']['attachments']) == 1:
                    photo_url = event.obj['message']['attachments'][-1]['photo']['sizes'][-1]['url']
                    count_likes_ayy = img_judge(url= photo_url, model=model_ayy)
                    count_likes_mdk = img_judge(url= photo_url, model=model_mdk)
                    send_message(vk_session, 'user_id', event.obj['message']['peer_id'], 'Хмм..\n Этот мем за 1000 просмотров соберет:\n {} лайков в паблике https://vk.com/dank_memes_ayylmao \n {} в https://vk.com/mudakoff'.format(int(count_likes_ayy[0][0]), int(count_likes_mdk[0][0]) ))
                    statuses_judge.update({event.obj['message']['peer_id']: [False, False]})
                    send_message(vk_session, 'user_id', event.obj['message']['peer_id'], 'Вот так)', keyboard=create_keyboard1())
                elif response == 'назад':
                    statuses_judge.update({event.obj['message']['peer_id']: [False, False]})
                    send_message(vk_session, 'user_id', event.obj['message']['peer_id'], 'Окей)', keyboard=create_keyboard1())
                else: #event.obj['message']['text'] and response != 'оценить' and response != 'назад':
                    send_message(vk_session, 'user_id', event.obj['message']['peer_id'], 'Мне нужна фотография, чтобя я смог ее оценить, \n или нажми кнопку чтобы вернуться или просто напиши "назад"', keyboard=create_keyboard2())


            if response == 'оценить' and not statuses_judge[event.obj['message']['peer_id']][0]:
                send_message(vk_session, 'user_id', event.obj['message']['peer_id'], 'Пришли мне фотографию, чтобы я оценил ее \n или нажми кнопку чтобы вернуться или просто напиши "назад"', keyboard=create_keyboard2())
                statuses_judge.update({event.obj['message']['peer_id']: [True, False]})

            if statuses_judge[event.obj['message']['peer_id']][1]:
                if len(event.obj['message']['attachments']) == 2:
                    url_1 = event.obj['message']['attachments'][0]['photo']['sizes'][-1]['url']
                    url_2 = event.obj['message']['attachments'][1]['photo']['sizes'][-1]['url']
                    core_img_path = tf.keras.utils.get_file('First.jpg', url_1)
                    backgr_img_path = tf.keras.utils.get_file('Second.jpg', url_2)
                    core_image = load_img(core_img_path)
                    backgr_image = load_img(backgr_img_path)
                    stylized_image = hub_module(tf.constant(core_image), tf.constant(backgr_image))[0]
                    ans = tensor_to_image(stylized_image)
                    os.remove('C:/Users/bushanka/.keras/datasets/First.jpg')
                    os.remove('C:/Users/bushanka/.keras/datasets/Second.jpg')
                    ans.save('Attach.jpg')
                    attachments = []
                    upload_image = upload.photo_messages(photos='Attach.jpg')[0]
                    attachments.append('photo{}_{}'.format(upload_image['owner_id'], upload_image['id']))
                    send_message(vk_session, 'user_id', event.obj['message']['peer_id'], attachment=attachments)
                    statuses_judge.update({event.obj['message']['peer_id']: [False, False]})
                    send_message(vk_session, 'user_id', event.obj['message']['peer_id'], 'Держи)', keyboard=create_keyboard1())
                elif response == 'назад':
                    statuses_judge.update({event.obj['message']['peer_id']: [False, False]})
                    send_message(vk_session, 'user_id', event.obj['message']['peer_id'], 'Окей)', keyboard=create_keyboard1())

                else: #event.obj['message']['text'] and response != 'стиль' and response != 'назад':
                    send_message(vk_session, 'user_id', event.obj['message']['peer_id'], 'Мне нужны две фотографии, чтобя я смогла поменять стиль', keyboard=create_keyboard2())
            if response == 'стиль' and not statuses_judge[event.obj['message']['peer_id']][1]:
                send_message(vk_session, 'user_id', event.obj['message']['peer_id'], 'Пришли мне 2 фотографии: первую - стиль которой хочешь поменять и вторую - стиль которой хочешь применить на первую фотографию \n или нажми кнопку чтобы вернуться или просто напиши "назад"', keyboard=create_keyboard2())
                statuses_judge.update({event.obj['message']['peer_id']: [False, True]})

            if response == 'привет' and not statuses_judge[event.obj['message']['peer_id']][0] and not statuses_judge[event.obj['message']['peer_id']][1]:
                send_message(vk_session, 'user_id', event.obj['message']['peer_id'], 'Здравствуй!')

            if response == 'помощь' and not statuses_judge[event.obj['message']['peer_id']][0] and not statuses_judge[event.obj['message']['peer_id']][1]:
                send_message(vk_session, 'user_id', event.obj['message']['peer_id'], 'Более подробную информацию читай тут!')

            if response == 'как дела?' and not statuses_judge[event.obj['message']['peer_id']][0] and not statuses_judge[event.obj['message']['peer_id']][1]:
                send_message(vk_session, 'user_id', event.obj['message']['peer_id'], 'Отлично как обычно')
