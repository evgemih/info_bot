# Телеграм-бот для накопления и поиска сведений.


## Запуск
```
python app.py
```

## Сборка и запуск docker контейнера
```
docker build --build-arg TG_BOT_TOKEN="ТОКЕН_ВАШЕГО_БОТА" . -t style_bot
```
В таком случае токен будет сохранен в контейнере.
Можно также при сборке не указывать токен, а передавать его при запуске контейнера:
```
docker run --name style_bot -d --rm -e TG_BOT_TOKEN="ТОКЕН_ВАШЕГО_БОТА" style_bot
```

## requirements.txt

Используются следующие библиотеки:
```
aiogram==3.3.0
numpy
torch
typing_extensions
collections
razdel
pathlib
pydantic
```

## Репозиторий содержит:

| | Описание файлов и папок |
| --- | --- |
| app.py | основной цикл программы, "точка входа"|
| Dockerfile | скрипт сборки докер контейнера |
| requirements.txt | необходимые библиотеки |
| data\ | Содержит файлы «базы знаний» |
| log\ | Содержит файлы журналов работы системы |
| handlers\default_commands.py | обработчики основных команд бота |
| handlers\dialog.py | обработчики диалога |
| expert\system.py | Классы и функционал ИС |


## Контакты

Телеграм: [@evyakuba](https://t.me/evyakuba)

## Ссылки

В освоении aiogram очень помог обучающий курс MasterGroosha [Пишем Telegram-ботов с aiogram 3.x](https://mastergroosha.github.io/aiogram-3-guide/quickstart/)
