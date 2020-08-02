import vk_api
from datetime import datetime
from vk_api.longpoll import VkLongPoll, VkEventType
from token_container import get_token as gt

my_token = gt()
vk_session = vk_api.VkApi(token= my_token)

session_api = vk_session.get_api()
longpoll = VkLongPoll(vk_session)

while True:
    for event in longpoll.listen():
        if event.type == VkEventType.MESSAGE_NEW:
            print('Get message in {}'.format(datetime.now()))
            print('Message_text: {}'.format(event.text))
            response = event.text.lower()
            if event.from_user and not event.from_me:
                if response == 'привет':
                    vk_session.method('messages.send', {'user_id': event.user_id, 'message': 'Привет!', 'random_id': 0})
                if response == 'отключись!':
                    break
