#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Настройки
"""
from pathlib import Path
from pydantic import SecretStr
from pydantic_settings import BaseSettings
from typing import List
from datetime import timedelta


BASE_DIRECTORY = Path(__file__).absolute().parent.parent


class Settings(BaseSettings):
    """Общие настройки."""

    # значения будут прочитаны из переменной окружения
    tg_bot_token: SecretStr = 'TOKEN'
    data_path: Path = BASE_DIRECTORY / 'data/'
    log_path: Path = BASE_DIRECTORY / 'log/'
    diff_path: Path = BASE_DIRECTORY / 'settings/diff.txt'

    SEP_TIME_DELTA: timedelta = timedelta(minutes=1)
    PARTS_SIMILARITY_THRESHOLD: float = 0.84
    OPTIONS_DIFF_THRESHOLD: float = 0.9
    MAX_OPTIONS: int = 6  # максимально число выдаваемых ответов для выбора
    MAX_ALTERNATIVES: int = 6  # число альтернативных смысловых блоков для поиска
    DIFFERENCE_THRESHOLD: float = 0.05
    #model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8')


config = Settings()
# config.tg_bot_token = os.environ.get("TG_BOT_TOKEN")

class DifferenceList(BaseSettings):
    sample: List[List[str]]

# Пример фраз, близких по сходству, но которые необходимо различать
diff_list = DifferenceList.parse_file(path = config.diff_path)
