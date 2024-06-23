# -*- coding: utf-8 -*-
"""
Info bot

@author: evgemih
"""

import logging
from datetime import datetime
from pathlib import Path
from aiogram import types, F, Router, html, flags
from aiogram.enums import ParseMode
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import StatesGroup, State

from handlers import default_commands
from expert.system import expert
from settings.base import config

router = Router()


# Определим класс состояний пользователя для отслеживания в диалоге
class UserStates(StatesGroup):
    start = State()
    input = State()
    explain = State()

# Функция для создания инлайн-клавиатуры
def get_inline_keyboard(data_dict) -> types.InlineKeyboardMarkup:
    buttons = []
    for k, v in data_dict.items():
        buttons.append([types.InlineKeyboardButton(
            text=k, 
            callback_data=v
            )])
    return types.InlineKeyboardMarkup(inline_keyboard=buttons)

    
def get_options_keyboard(options_list) -> types.ReplyKeyboardMarkup:
    button_data = {}
    for i, text in enumerate(options_list):
        button_data[text] = f'select_{i}'
    return get_inline_keyboard(button_data)


@router.message(F.text.lower() == "начать заново")
async def button_restart(message: types.Message, state: FSMContext) -> None:
    await default_commands.cmd_start(message, state)


@router.message(F.text.lower() == "помощь")
async def button_help(message: types.Message, state: FSMContext) -> None:
    await default_commands.cmd_help(message, state)


# Обработчик начального состояния
@router.message(UserStates.start)
async def start_state(message: types.Message, state: FSMContext) -> None:
    logging.debug('Обработка start_state')
    expert.reset()
    await state.set_state(UserStates.input)
    # выводим сообщение и прикрепляем инлайн-клавиатуру к нему
    kb = types.ReplyKeyboardRemove()
    await message.answer(text="Введите проблему:", reply_markup = kb)


# Обработчик состояния вывода объяснения
@router.message(UserStates.explain)
async def explain_state(message: types.Message, state: FSMContext) -> None:
    logging.debug('Обработка explain_state')
    await state.set_state(UserStates.input)
    await message.answer(
        expert.get_explain(), 
        parse_mode=ParseMode.HTML, 
        )
        

# Если нажата одна из кнопок, где data начинается на "select_"
@router.callback_query(F.data.startswith("select_"))
async def msg_change(callback: types.CallbackQuery, state: FSMContext) -> None:
    # определяем номер ответа, разделив по знаку _
    select_n = int(callback.data.split("_")[1])
    option_text = callback.message.reply_markup.inline_keyboard[select_n][0].text
    logging.info('Выбрано -> ' + option_text)

    msg_text = callback.message.text.strip()
    # Если пользователь ввел запрос без точки, запятой или вопроса в конце
    if not msg_text.endswith(('.',',','?')):
        # добавим точку, чтобы система могла в дальнейшем разделить сведения
        msg_text += '.'

    # Записываем на диск
    outfile = Path(config.data_path, datetime.now().strftime('%Y-%m-%d')+'.txt')
    try:
        with open(outfile, 'a', encoding='UTF-8') as f:  #,encoding='cp1251'
            f.write(' ' + option_text)
    except IOError as error:
        logging.error(error)

    # Добавляем в память
    expert.add_text_data(option_text)

        
    # Добавляем к тексту сообщения текст на нажатой кнопке
    new_msg_text = f"{msg_text} {option_text}"
    new_msg_text_html = f"{msg_text} {html.bold(html.quote(option_text))}"
    
    logging.debug('msg_change <<< ' + new_msg_text)
    try:
        data = expert.get_answer(new_msg_text)
    except Exception as e:
        logging.error(e)    
    
    kb = types.ReplyKeyboardRemove()
    if data:
        for item in data:
            options = item.get('options')
            await state.update_data(answer_options=options)
            if options:
                kb = get_options_keyboard(options)
    
    msg = await callback.message.edit_text(
        text = new_msg_text_html, 
        parse_mode = ParseMode.HTML,
        reply_markup = kb
    )
    await state.update_data(prev_msg=msg)
    await state.update_data(prev_datetime = datetime.now())
    await callback.answer()


# Обработка любого ввода пользователя, если не сработали обработчики выше
@router.message()
async def input_state(message: types.Message, state: FSMContext) -> None:
    # msg = await message.answer("⏳ обработка запроса...")
    logging.debug('input_state <<< ' + message.text)

    # user_name = message.chat.first_name + ' ' + message.chat.last_name
    user_name = message.from_user.full_name

    msg_text = message.text.strip()
    # Если пользователь ввел запрос без точки, запятой или вопроса в конце
    if not msg_text.endswith(('.',',','?')):
        # добавим точку, чтобы система могла в дальнейшем разделить сведения
        msg_text += '.'
    
    # Можно прочитать message.date, но он в формате UTС+0
    msg_datetime = datetime.now()
    str_datetime = msg_datetime.strftime('%d.%m.%Y %H:%M')
    # text_to_write = f'{user_name}, [{str_datetime}]\r\n{message.text}\r\n\r\n'
    text_to_write = f'\r\n\r\n{user_name}, [{str_datetime}]\r\n{msg_text} '

    # Записываем на диск введенную пользователем информацию
    outfile = Path(config.data_path, datetime.now().strftime('%Y-%m-%d')+'.txt')
    try:
        with open(outfile, 'a', encoding='UTF-8') as f:  #, encoding='cp1251'
            f.write(text_to_write)
    except IOError as error:
        logging.error(error)

    # Добавляем новую информацию в память системы
    expert.add_text_data(text_to_write)

    # Получем ответ
    try:
        data = expert.get_answer(message.text)
    except Exception as e:
        logging.error(e)
        data = None

    if not data:
        return await message.answer("🚫 Ошибка при обработке запроса.", reply_markup=None)
    
    # Перебираем полученные ответы
    for item in data:
        text = item.get('text')
        options = item.get('options')
        await state.update_data(answer_options=options)
        
        user_data = await state.get_data()            
        prev_msg = user_data.get('prev_msg')
        # Удаляем кнопки в предыдущем сообщении
        if prev_msg:
            try:
                await prev_msg.edit_text(text = prev_msg.text, reply_markup = None)
            except Exception as e:
                logging.error(e)
                
        # формируем новые кнопки
        kb = types.ReplyKeyboardRemove()
        if options:
            kb = get_options_keyboard(options)

        # выводим сообщение и прикрепляем инлайн-клавиатуру к нему
        msg = await message.answer(text, reply_markup = kb)
        await state.update_data(prev_msg=msg)

