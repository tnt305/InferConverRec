import re 

def is_valid_sentence(sentence):
    # Placeholder cho hàm is_valid_sentence
    # Thay đổi điều kiện theo nhu cầu của bạn
    return len(sentence.split()) < 10 and not any(char in sentence for char in ['.', ';'])


def rewrite(sentence):
    # Các danh từ và placeholder
    replacements = {
        r'\bit\b': '<movie>',          # Thay thế 'it' bằng <movie>
        r'\bthat movie\b': '<movie>',  # Thay thế 'that movie' bằng <movie>
        r'\bthis movie\b': '<movie>'   # Thay thế 'this movie' bằng <movie>
    }
    
    # Thay thế các danh từ theo quy định trong replacements
    for pattern, replacement in replacements.items():
        sentence = re.sub(pattern, replacement, sentence, flags=re.IGNORECASE)
    
    # Thêm <movie> vào sau các động từ cụ thể nếu chúng xuất hiện trước từ cần thay thế
    patterns = [r'\brecommend\b', r'\bsuggest\b', r'\boffer\b', r'\btry\b']
    for pattern in patterns:
        sentence = re.sub(rf'({pattern})\s+(\w+)', r'\1 <movie>', sentence, flags=re.IGNORECASE)
    
    return sentence

def rewrite2(sentence):
    verb_patterns = [
        r'recommend', r'suggest', r'offer', r'try', r'try out'
    ]
    
    # Kết hợp các mẫu động từ thành một biểu thức chính quy
    verb_pattern = r'|'.join(verb_patterns)
    
    # Tìm <|endoftext|> ở cuối câu và động từ ngay trước nó
    match = re.search(rf'({verb_pattern})\s*<\|endoftext\|>$', sentence)
    
    if match:
        # Nếu tìm thấy, thêm <movie> sau động từ và xóa <|endoftext|>
        verb = match.group(1)
        sentence = re.sub(rf'{verb}\s*<\|endoftext\|>$', f'{verb} <movie>', sentence)
    else:
        # Nếu không tìm thấy, chỉ xóa <|endoftext|> nếu nó ở cuối câu
        sentence = re.sub(r'<\|endoftext\|>$', '', sentence)
    
    # Loại bỏ khoảng trắng thừa
    sentence = re.sub(r'\s+', ' ', sentence).strip()
    
    return sentence



