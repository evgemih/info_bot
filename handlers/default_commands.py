from aiogram import Bot, Router, types
from aiogram import html
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext

from handlers.dialog import start_state, explain_state

router = Router()

system_menu = {
    'start': 'Начать сначала', 
    'explain': 'Объясни последнее', 
    'help': 'Помощь'
    }

help_text = """
Информационная система по ремонту компьютеров.
      
Для начала работы, введите описание проблемы и известные сведения в одном сообщении.
Система найдет ниболее подходящие сведения и предложит несколько предложений, которые чаще всего встречаются рядом с введенными сведениями.
Выбирайте тот вариант, который подходит для текущей ситуации.

Для добавления новых данных в систему, напишите в одном сообщении описание проблемы и все необходимые сведения для решения проблемы.
Каждое сообщение записывается в базу данных и используется в дальнейшем для поиска информации.

Используйте предельно короткие и четкие предложения. Одна мысль - одно предложение.

Для вывода объяснения, выберите в меню соответствующий пункт.
"""

async def set_system_commands(bot: Bot):
    # устанавливаем меню по-умолчанию
    commands = []
    for c, d in system_menu.items():
        commands.append(types.BotCommand(command=c, description=d))
 
    # для каждого чата (и пользователя) можно установить свое меню
    await bot.set_my_commands(commands, types.BotCommandScopeDefault())


# Чтобы функция стала обработчиком события, нужно
# оформить ее через специальный декоратор
# или зарегистрировать ее у диспетчера или роутера

# Обработчик события (хэндлер) на команду /start
@router.message(Command("start"))
async def cmd_start(message: types.Message, state: FSMContext):
    # Определяем имя пользователя и выводим приветствие
    await message.answer(
        f"Привет, {html.bold(html.quote(message.from_user.full_name))}!",
        parse_mode=ParseMode.HTML,
        reply_markup=None
    )
    # и переходим в начальное состояние
    await start_state(message, state)


# Хэндлер на команду /explain для вывода объяснения 
@router.message(Command("explain"))
async def cmd_explain(message: types.Message, state: FSMContext):
    await explain_state(message, state)


# Хэндлер на команду /help
@router.message(Command("help"))
async def cmd_help(message: types.Message, state: FSMContext):
    await message.answer(help_text)

