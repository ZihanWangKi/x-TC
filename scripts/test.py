

with open("prompt.txt", mode='r', encoding='utf-8') as text_file:
    x=text_file.read()
print(repr(x)[1:-1])