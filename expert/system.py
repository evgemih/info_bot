# -*- coding: utf-8 -*-
"""
@author: evgemih
"""

# %% Импорт библиотек

import numpy as np
import torch
import logging
from collections import Counter
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
from razdel import sentenize  # tokenize
from pathlib import Path

from settings.base import config, diff_list

# %% Загрузка модели
logging.debug('Загружаем токенайзер')
tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
logging.debug('Загружаем модель')
model = AutoModel.from_pretrained("cointegrated/rubert-tiny2")
# self.model.cuda()  # uncomment it if you have a GPU

# %% Глобальные переменные
EMB_SIZE = 312  # размер эмбеддингов
SEP_TOKEN = '[SEP]'
EXIT_CMD = 'exit'

# Список разрешенных символов, все остальные будут заменены пробелами
CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHARS+= "абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
CHARS+= "!\"#$%&'()*+,-./0123456789:;<=>?@[\]^_`{|}~\n"

# Список стоп-слов, которые будут удалены
stopwords = set([])
# Словарь замены подстрок, внимательно с этим!
replacement_dict = {
    "хдд" :"hdd",
}
# Словарь замены с помощью регулярных выражений
# пока не требуется
# regular_exp_dict = {
    # '\.+'  :'.',   # замена многоточия на .
# }


def preprocess_text(text):
    """Предобработка текста."""    
    # переведем в нижний регистр
    text = text.lower()
    # Произведем замены по словарю replacement_dict
    for key, value in replacement_dict.items():
        text = text.replace(key, value)
    # Произведем замены по словарю regular_exp_dict
    # for key, value in regular_exp_dict.items():
    #     text = re.sub(key, value, text)
    # Оставим в предложении только символы из CHARS
    filtered_ch_list = []
    for ch in text:
        filtered_ch_list.append(ch if ch in CHARS else ' ')

    return ''.join(filtered_ch_list)


def sentenize_text(text):
    """Разбиение текста на предложения."""
    sents = sentenize(text)
    return [s.text for s in sents]


def embed_bert_cls(text, model, tokenizer):
    """Получаем эмбединги из text, с помощью model и tokenizer."""
    # t = preprocess_text(text)
    t = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**{k: v.to(model.device) for k, v in t.items()})
    embeddings = model_output.last_hidden_state[:, 0, :]
    embeddings = torch.nn.functional.normalize(embeddings)
    return embeddings[0].cpu().numpy()


def read_text_from_file(filename, encodings=['windows-1251', 'utf-8']):
    """Чтение файла с перебором заданных кодировок."""
    for e in encodings:
        error = None
        try:
            with open(filename, 'r', encoding=e) as f:
                t = f.read()
                return t, error
        # Если файл в неправльной кодировке, сработает исключение
        except UnicodeDecodeError:  # as error
            continue  # и тогда попробуем открыть с другой кодировкой
        except IOError as error:
            # print(error)
            return t, error

    return None, 'Not find the correct encoding'


def get_data(text):
    return Fragment(text, embed_bert_cls(text, model, tokenizer))


def get_data_list(texts_list):
    res = []
    for text in texts_list:
        res.append(get_data(text))
    return res


def get_similarity_list(embedding, data_list, logging=False): 
    """Получение списка схожести embedding с data_list."""
    sim = []
    for i in range(len(data_list)):
        sim.append(np.sum(embedding * data_list[i].emb))
        if logging:
            print(f'{i} {sim[-1]:.4f} {data_list[i].text}')
    return sim


def get_emb(text):
    """Получение эмбеддингов текста."""
    return embed_bert_cls(text, model, tokenizer)
    

class Fragment:
    """Текстовый фрагмент с эмбеддингами."""
    def __init__(self, text, embeddings = []):
        self.text = text
        self.emb = embeddings


class ExpertSystem:
    def __init__(self, database_root, model, tokenizer):

        self.database_root = database_root
        self.blocks_data = [[]]      
        self.parts_text = [[]]
        self.input_text = ''
        self.input_parts = []
        self.sim_blocks_data = []
        
        self.tokenizer = tokenizer
        self.model = model
        # self.model.cuda()  # uncomment it if you have a GPU
        
        self.exit = False
        self.last_input_time = datetime.now()
        self.explain_text = ''        

        self.load_data(database_root)


    def reset(self):
        logging.debug('Перезагрузка информационной системы.')
        self.__init__(self.database_root, self.model, self.tokenizer)


    def add_text_data(self, text):
        # logging.debug('Добавление сведений в память.')
        
        for line in text.splitlines():
            line = line.strip()  # удаляем пустые символы

            # Находим строки вида 
            # Имя пользователя, [31.05.2024 12:00]

            idx1 = line.find(", [")
            if idx1!=0 and len(line)>19 and line[-1]==']': 
                # user_name = line[:idx1]
                s = line[-17:-1]
                msg_datetime = datetime.strptime(s, '%d.%m.%Y %H:%M')
                # Если разница во времени между сообщениями больше порога
                if (msg_datetime - self.last_input_time) >= config.SEP_TIME_DELTA:
                    # logging.debug('Добавляем новый смысловой блок.')
                    self.parts_text.append([])
                    self.blocks_data.append([])
                
                # фиксируем время последней введенной информации.                    
                self.last_input_time = msg_datetime
            elif len(line) > 0:
                # блок может содержать несколько разных сообщений
                self.parts_text[-1].extend(sentenize_text(line))
                self.blocks_data[-1] = get_data(' '.join(self.parts_text[-1]))
                # logging.debug(self.blocks_data[-1].text)


    def load_data(self, data_path):
        """Загрузка данных из файлов."""
        for filename in Path(data_path).rglob('*.txt'):
            logging.debug(f'Загрузка: {filename}')
            text, error = read_text_from_file(filename)
            if (error):
                self.error = error
                logging.error(error)
                return False
    
            self.add_text_data(text)


    def process_input(self, input_text=''):
        """Обработка ввода."""        
        # Для работы из терминала, раскомментировать:
        # if input_text:
        #     print(input_text)
        # else:
        #     input_text = input()
        self.input_text = input_text
        self.last_input_time = datetime.now()
        # иногда из концоли захватывает не только что ввели, но и вот это:
        logging.info('=====================================================')
        logging.info(f'ВВОД <<< {input_text}')
        logging.info('=====================================================')        
        self.need_input = False
        
        if (input_text.lower() == EXIT_CMD):
            self.exit = True
            return

        self.input_parts = []
        # разделение текста на предложения
        for part in sentenize_text(input_text):
            self.input_parts.append(part)
        
        self.input_data = get_data_list(self.input_parts)
        
        return #text


    def get_explain(self):
        """Получение объяснения."""
        logging.info(self.explain_text)
        return self.explain_text


    def get_answer(self, text):
        """Получение ответа системы."""
        self.process_input(text)
        input_data = self.input_data
        
        # Пример фраз, близких по сходству, но которые необходимо различать
        # вычислить разницу между фразами, если несколько пар - взять среднее
        
        diff_data = []
        for pair in diff_list.sample:
            pair_data = get_data_list(pair)
            pair_data[2].emb = pair_data[1].emb - pair_data[0].emb
            diff_data.append(pair_data[2])

        # Составляем словарь фраз 
        # и вычисляем частоту встречаемости фраз только среди отобранных блоков

        vocab_data = get_data_list([SEP_TOKEN])
        parts_count = Counter()
        part_to_vocab = {}

        # Из запроса тоже добавим фразы
        # Добавляем фразы из запроса в обратном порядке для того, чтобы последняя
        # введенная фраза первой попадала в словарь. Например если вводим противоположные
        # факты то последний введенный будет зафиксировать, а предыдущий будет пропущен
        logging.info('Формируем словарь из предложений. В начало добавляем предложения из запроса:')
        for part in reversed(input_data):

            vocab_data.append(part)
            parts_count[part] += 1  # нужно ли добавлять счетчик?
            part_to_vocab[part] = part
            logging.info(f'+ {part.text}') #  с {fragments_data[max_idx].text}

        input_tokens=[]
        for part in input_data:
            input_tokens.append(part_to_vocab[part])


        logging.info('Перебираем блоки, похожие на запрос:')
        
        inp_emb = get_emb(self.input_text)
        blocks_sim = get_similarity_list(inp_emb, self.blocks_data, logging=False)

        # Задаем число альтернативных версий
        # чем больше число, тем более верная статистика ?
        # тем более надежна "логика"
        n = config.MAX_ALTERNATIVES
        
        sorted_sim = np.argsort(blocks_sim) #[::-1][:n]
        
        block_data = []
        for block_id in reversed(sorted_sim):
            logging.info('===========================')
            logging.info(f'{blocks_sim[block_id]:.2f} {self.blocks_data[block_id].text}')
            block = get_data_list(self.parts_text[block_id])

            block_vocab_parts = []
            find_input_parts = False
            for part in block:
                sims = get_similarity_list(part.emb, vocab_data, logging=False)
                max_idx = np.argmax(sims)
                # сравнение частей между собой, оставляем только различные по смыслу        
                if sims[max_idx] < config.PARTS_SIMILARITY_THRESHOLD:
                    block_vocab_parts.append(part)
                    part_to_vocab[part] = part
                    logging.info(f'{sims[max_idx]:.2f} + "{part.text}"') #  с {fragments_data[max_idx].text}
                else:
                    logging.info(f'{sims[max_idx]:.2f}>{config.PARTS_SIMILARITY_THRESHOLD} "{part.text}" сходство с "{vocab_data[max_idx].text}"')
                    find_part = vocab_data[max_idx]
                    for dif in diff_data:
                        dif_sims = get_similarity_list(dif.emb, [part, vocab_data[max_idx]], logging=False)
                        dif_delta = max(dif_sims) - min(dif_sims)
                        if dif_delta > config.DIFFERENCE_THRESHOLD:
                            find_part = part
                            logging.info(f'Найдено отличие: {dif.text} ({dif_delta:.2f})')
                            logging.info(f'{sims[max_idx]:.2f} + {part.text}') #  с {fragments_data[max_idx].text}
                            break

                    block_vocab_parts.append(find_part)        
                    if find_part in input_tokens:
                        find_input_parts = True
                    
                    part_to_vocab[part] = find_part
                    
            if find_input_parts:
                n -= 1
                logging.info('+++ Добавляем блок и пересчитываем статистику.')
                block_data.append(block)
                for part in block_vocab_parts:
                    vocab_data.append(part)
                    parts_count[part] += 1
            else:
                logging.info('--- Не найдены предложения из запроса, пропускаем блок.')
                    
        
            if n == 0:
                break
        
        if len(block_data)==0:
            logging.info('Не найдено подходящих данных!')

        self.sim_blocks_data = block_data
        
        parts_count[vocab_data[0]]=0  # обнуляем счетчик для [SEP]

        logging.info('Самые часто встречающиеся фразы:')
        for part, cnt in parts_count.most_common(10):
            logging.info(f'{cnt} {part.text}')
                
        # Если в блоке есть фразы, которые похожи на фразы из запроса и 
        # отличаются по какому-то свойству согласно списка отличий
        # тогда, нужно раздельно считать частоту их встречаемости
        
        self.explain_text = 'Найдены близкие по содержанию сведения:'
        fragments_data = input_data.copy()
        logging.info('Получение списка подходящих фраз:')
        for block_idx, block in enumerate(self.sim_blocks_data):
            logging.info('===========================')
            
            # Самая высокочастотная фраза в блоке, не совпадающая с входными
            target_part_idx = len(block)-1  # по умолчанию цель - последняя фраза
            target_part_count = 0
        
            part_to_input = {}
            part_is_input = {}
            # устанавливаем соответствие с фразами из запроса
            # ищем самую высокочастотную фразу не совпадающую с запросом
            for part in block:
                token = part_to_vocab[part]
                if token in input_tokens:
                    in_idx = input_tokens.index(token)
                    part_to_input[part] = in_idx
                    part_is_input[part] = True
                else:
                    part_is_input[part] = False
                    if target_part_count < parts_count[token]:
                        target_part_count = parts_count[token]
                        target_part_idx = block.index(part)
                        
        
            # print('Цель: ', block[target_part_idx].text)
            
            # Если целевая фраза уже есть в введенных данных
            # найти следующую фразу начиная от уже введенных
            # по направлению к целевой
        
            # Вычислим расстояние между целевой фразой и каждой из введенных
            to_target_offset = []
            min_positive_dist = len(block)
            logging_text = ''
            for part_idx, part in enumerate(block):
                part_text = part.text
                dist = len(block)
                if part in part_to_input:
                    part_text = f'[{part_text}]'
                    dist = target_part_idx - part_idx  # было abs()
                    if 0<=dist and dist < min_positive_dist:
                        min_positive_dist = dist

                if part_idx == target_part_idx:
                    dist = 0      
                    part_text = f'-->{part_text}<--'
                
                to_target_offset.append(dist)
                logging_text += part_text + ' '
            
            logging.info(logging_text)
          
            # Определяем предложение для вывода пользователю
            next_part = None
            for i in np.argsort(to_target_offset):
                offset = to_target_offset[i]
                # Если целевая фраза находится слева
                if offset < 0:  
                    continue  # пропускаем
                # Если выходим за границы блока
                elif offset >= len(block):
                    continue  # пропускаем
                # Если целевая фраза находится справа
                elif offset > 0:
                    # берем следующую фразу
                    next_idx = target_part_idx - offset + 1
                    if next_idx >= len(block):
                        next_idx = len(block)
                # Если совпадает с целевой, то берем целевую
                else:  
                    next_idx = target_part_idx
        
                next_part = block[next_idx]

                if next_part in part_to_input:
                    # print('(совпадает с введенными данными, пропускаем)')
                    continue
            
                # Вычисляем сходство полученной next_part с уже ранее добавленными
                sims = get_similarity_list(next_part.emb, fragments_data, logging=False)
                max_idx = np.argmax(sims)
                
                # Если сходство по смыслу меньше заданного порога
                if sims[max_idx] < config.OPTIONS_DIFF_THRESHOLD:
                    # добавляем в список
                    fragments_data.append(next_part)
                    logging.info(f'+ {next_part.text} (max_s:{sims[max_idx]:.2f}<{config.OPTIONS_DIFF_THRESHOLD})') #  с {fragments_data[max_idx].text}
                    # для добавляем только одной фразы с каждого блока
                    # break  # расскомментировать
                
                # Если фразы близки по смыслу
                else:
                    logging.info(f'- {next_part.text} сходство с {fragments_data[max_idx].text} ({sims[max_idx]:.2f}>{config.OPTIONS_DIFF_THRESHOLD})')
                    # Проверяем наличие заданных "отличительных признаков"
                    for dif in diff_data:
                        dif_sims = get_similarity_list(dif.emb, [next_part, fragments_data[max_idx]], logging=False)
                        dif_delta = max(dif_sims) - min(dif_sims)
                        # Если отличие выше порога, заданного в настройках
                        if dif_delta > config.DIFFERENCE_THRESHOLD:
                            # значит фразы отличаются по этому признаку
                            # добавляем в список
                            fragments_data.append(next_part)
                            logging.info(f'Найдено отличие: {dif.text} ({dif_delta:.2f})')
                            logging.info(f'+ {next_part.text}')
                            break
                    
            # Формируем текст "объяснения"
            self.explain_text += '\n\n'
            for part_idx, part in enumerate(block):
                part_text = part.text
                 # TODO добавить экранирование HTML тегов
                if part in part_to_input:
                    part_text = '<i>'+part_text+'</i>'
                if part_idx == target_part_idx:
                    part_text = '<u>'+part_text+'</u>'
                if part in fragments_data:
                    part_text = '<b>'+part_text+'</b>'
                self.explain_text += part_text+' '
            
        self.explain_text += (
            "\n\nОбозначения: <i>курсивом</i> выделено совпадение с запросом, "
            "<u>подчеркнуто</u> - целевое предложение, "
            "<b>жирным</b> обозначены предложения для выбора пользователем, ближайшие к цели от уже известных сведений."
        )

        
        # удаляем добавленные в начало фразы из запроса
        del(fragments_data[:len(input_data)])
        
        logging.info('=====================================================')
        logging.info(text)
        logging.info('====== Список для вывода: ======')
                
        # Формируем окончательный ответ системы
        # количество предлагаемых вариантов ограничено config.MAX_OPTIONS
        answer_options = []
        for part in fragments_data[:config.MAX_OPTIONS]:
            answer_options.append(part.text)
            logging.info(f'{part.text}')

        return [{
            'text':text,
            'options':answer_options,
            }]


logging.debug('Инициализируем ExpertSystem')
expert = ExpertSystem(config.data_path, model, tokenizer)
