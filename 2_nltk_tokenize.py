
import glob

if __name__ == '__main__':
    folder = 'test'
    
    filenames = glob.glob(folder + '/*.txt')
    for fname in filenames:
        print(fname)


