import os

resource_path = 'live/'

start = 20000
num_of_zeros = 3

files = os.listdir(resource_path)

for file in files:
    if not file.endswith('.jpg'):
        os.remove(resource_path+file)
    elif ('media_id=' in file) & file.endswith('.jpg'):
        os.remove(resource_path+file)

files = os.listdir(resource_path)

for file in files:
    if file.endswith('.jpg'):
        os.rename(resource_path+file, resource_path+'{}'.format('0'*num_of_zeros)+str(start)+'.jpg')
        start = start + 1
