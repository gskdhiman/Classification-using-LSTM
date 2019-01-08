import os

'''create a directory wit the name of Google_News_vectors'''
dirpath =os.path.join('Google_News_vectors')
if not os.path.exists(dirpath):
    os.mkdir(dirpath)
    
print("Download vectors from the following location")
print(r'https://drive.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM&export=download')
print("Place them in the given diretory",os.path.abspath(dirpath))

while(True):
    inp = input("Have you done the above step: (y/n)")
    if inp.lower() == 'y':
        dataset_path =os.path.join('Dataset')
        if not os.path.exists(dataset_path):
            os.mkdir(dataset_path)
        
        print("Downloading dataset from the following location")
        print(r'https://www.cs.umb.edu/~smimarog/textmining/datasets/r8-train-stemmed.txt')
        !curl https://www.cs.umb.edu/~smimarog/textmining/datasets/r8-train-stemmed.txt > Dataset//r8-train-stemmed.txt
        !curl https://www.cs.umb.edu/~smimarog/textmining/datasets/r8-test-stemmed.txt > Dataset//r8-test-stemmed.txt
        print("the datasets have been downloaded and have been placed in the diretory: ",os.path.abspath(dataset_path))
        break
    else:
        print('Do the above steps then proceed further')