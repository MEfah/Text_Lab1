import re
from argparse import ArgumentParser, Namespace
from pathlib import Path
from nltk import tokenize
from nltk.corpus import stopwords
import string
from nltk import download
from nltk import stem
from pymorphy3 import MorphAnalyzer
import time
from typing import Callable


def get_args() -> Namespace:
    argparser = ArgumentParser(prog = 'text_processor')
    default_path = Path() / 'texts'
    argparser.add_argument('-f', '--from_path', default=str(default_path), required=False)
    argparser.add_argument('-t', '--to_path', default=str(default_path / 'output' / 'output.txt'), required=False)
    argparser.add_argument('-c', '--clear', default=False, required=False, action='store_true')
    return argparser.parse_args()
    

def clear_text_from_nums(path: Path) -> str:
    for file_path in path.glob('*.txt'):
        text = file_path.read_text(encoding='utf-8')
        text = re.sub('\(\d+\)', '', text)
        file_path.write_text(text, encoding='utf-8')
        
        
def process_tokens(tokens: list, l: Callable[[str], str]) -> dict:
    word_dict = {}
    for token in tokens:
        m = l(token)
        if m in word_dict.keys():
            word_dict[m] = word_dict[m] + 1
        else:
            word_dict[m] = 1
    return word_dict


def main():
    download('stopwords')
    download('punkt')
    
    args = get_args()
    from_path = Path(args.from_path)
    to_path = Path(args.to_path)
    
    if not from_path.is_dir(): from_path.mkdir()
    if not from_path.is_file(): from_path.touch()
    
    if args.clear:
        clear_text_from_nums(from_path)
    
    output = ''
    for file_path in from_path.glob('*.txt'):
        text = file_path.read_text(encoding='utf-8')
        tokens = tokenize.word_tokenize(text, language='russian')
        stop_words = stopwords.words('russian')
        tokens = [token.lower() for token in tokens if token not in (string.punctuation + '...«»–') and token not in stop_words]
        
        start_time = time.time()
        morph_analyzer = MorphAnalyzer(lang='ru')
        word_dict = process_tokens(tokens, lambda x: morph_analyzer.parse(x)[0].normal_form)
        output = output + '\n\n\n' + file_path.name + ' Лемматизация\n' + re.sub('(?<=\)),', '\n', str(sorted(word_dict.items(), key=lambda x: x[1], reverse=True))) 
        output = output + '\nВремя выполнения в секундах: ' + str(time.time() - start_time)
        
        start_time = time.time()
        stemmer = stem.SnowballStemmer(language='russian')
        word_dict_stem = process_tokens(tokens, lambda x: str(stemmer.stem(x)))
        output = output + '\n\n\n' + file_path.name + ' Стемминг\n' + re.sub('(?<=\)),', '\n', str(sorted(word_dict_stem.items(), key=lambda x: x[1], reverse=True)))
        output = output + '\nВремя выполнения в секундах: ' + str(time.time() - start_time)

    to_path.write_text(output.strip(), encoding='utf-8')
        

if __name__ == '__main__':
    main()
