import os
count = 1


def create(file_name):
    f = open(f'{file_name}.txt', 'w')
    f.close()


def write(file_name):
    lines = int(input('How many lines? >>'))
    f = open(f'{file_name}.txt', 'w')

    for i in range(lines):
        f.writelines(input(f'Input message for line {i+1}: '))
        f.writelines('\n')
    f.close()


def read(file_name):
    f = open(f'{file_name}.txt', 'r')
    for l in f:
        print(l)
    f.close()


def destroy(file_name):
    if os.path.exists(f'{file_name}.txt'):
        os.remove(f'{file_name}.txt')
        print('File is deleted.')
    else:
        print('File does not exist.')

def chkFile(file_name):
    if os.path.exists(f"{file_name}.txt"):
        return True
    else:
        print("File does not exists.")
        return False