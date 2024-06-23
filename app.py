# -*- coding: utf-8 -*-
"""
Info bot

@author: evgemih
"""

import os
from aiogram import Bot, Dispatcher
import asyncio
import logging


from settings.base import config
from handlers import default_commands, dialog


# Включаем логгирование
logging.root.handlers = []
logging.basicConfig(level=logging.DEBUG,
                    filename=os.path.join(config.log_path, "main.log"),
                    filemode="w",  # a - добавлять
                    format="%(asctime)s %(levelname)s %(message)s")
# logging.getLogger().addHandler(logging.StreamHandler())  # вывод лога в концоль

# Объект бота
bot = Bot(config.tg_bot_token.get_secret_value())

# Диспетчер
dp = Dispatcher()

# Подключаем обработчики событий
dp.include_router(default_commands.router)
dp.include_router(dialog.router)


async def main():
    await default_commands.set_system_commands(bot)
    # Удалим все накопленные события в очереди
    await bot.delete_webhook(drop_pending_updates=True)
    # Процесс постоянного опроса сервера на наличие новых событий от Telegram
    # При поступлении события, диспетчер вызовет первую(!) подходящую
    # по всем фильтрам ф-ю обработки
    print("Run bot")
    await dp.start_polling(bot)

# Запуск основного цикла программы
if __name__ == "__main__":
    asyncio.run(main())
    logging.debug('========== EXIT ==========')
