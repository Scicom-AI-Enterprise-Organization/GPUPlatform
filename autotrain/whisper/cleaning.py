import re

def fix_spacing(text):
    quote_pattern = r'"([^"]*)"'
    def fix_quotes(match):
        content = match.group(1).strip()
        return f'"{content}"'

    text = re.sub(quote_pattern, fix_quotes, text)

    paren_pattern = r'\(([^)]*)\)'
    def fix_parens(match):
        content = match.group(1).strip()
        return f'({content})'

    text = re.sub(paren_pattern, fix_parens, text)
    text = re.sub(r'\s+([,\.!?])', r'\1', text)
    return text

def whisper_textcleaning(text):
    text = re.sub(r'\[.*?\]|\(.*?\)', '', text)
    text = re.sub(r'\b(?:ok|okay)\b', 'OK', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(h+m+|erm+|mm+|uhh+)\b', 'herm', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(a+h+|a+\s*a+)(?=[\s,\.!?]|$)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\.{2,}', ',', text)
    text = re.sub(r'(?<=\s)-(\w+)\b', r'\1', text)
    text = re.sub(r'\b(\w+)-(?=\s|$)', r'\1', text)
    text = re.sub(r'\b(um|uh|aa|erm)(\s+\1)+', r'\1', text, flags=re.IGNORECASE)
    text = re.sub(r'([.,!?])(?=[^\s])', r'\1 ', text)
    text = fix_spacing(text)
    text = re.sub(r'\s+', ' ', text).strip()
    if len(text) and text[0] in ',.?:':
        text = text[1:].strip()
        text = re.sub(r'\s+', ' ', text).strip()
    return text