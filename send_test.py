from telethon import TelegramClient
from telethon.errors import UsernameNotOccupiedError, PeerIdInvalidError, UserPrivacyRestrictedError
import asyncio

api_id = 21409208
api_hash = 'ea7a50ff782cacf103cfb3636914f224'
session_name = 'user_session'

FILENAME = 'logs/links/target_nicknames.txt'

def generate_message(username):
    return (f"🧠 Привет! Я делаю диплом по визуальному восприятию и собираю анонимные ответы. Нужно просто закрасить картинку. Это не тест и не оценка, просто исследование) Можешь помочь? Вот ссылка 👉 https://diplomissledovanie.onrender.com/?user={username.lstrip('@')}")

async def send_links():
    client = TelegramClient(session_name, api_id, api_hash)
    await client.start()

    with open(FILENAME, 'r', encoding='utf-8') as f:
        usernames = [line.strip() for line in f if line.strip()]

    for username in usernames:
        try:
            entity = await client.get_entity(username)
            if hasattr(entity, 'bot') and entity.bot:
                print(f"⛔ {username} — это бот. Пропущен.")
                continue

            await client.send_message(entity, generate_message(username))
            print(f"✅ Отправлено: {username}")

        except UsernameNotOccupiedError:
            print(f"❌ Не найден: {username}")
        except UserPrivacyRestrictedError:
            print(f"🔒 Ограничения приватности: {username}")
        except PeerIdInvalidError:
            print(f"⚠️ Нельзя отправить: {username}")
        except Exception as e:
            print(f"🔥 Ошибка у {username}: {e}")

    await client.disconnect()

if __name__ == '__main__':
    asyncio.run(send_links())
