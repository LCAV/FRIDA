import numpy as np
import os, sys, getopt

def create_edm(argv):

    input_file = ''
    output_dir = ''
    try:
      opts, args = getopt.getopt(argv,"hi:o:",["input_file=","output_dir="])
    except getopt.GetoptError:
      print 'create_matrix.py -i <input_file> -o <output_dir>'
      sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'create_matrix.py -i <input_file> -o <output_dir>'
            sys.exit()
        elif opt in ("-i", "--input_file"):
            txtfile = arg
        elif opt in ("-o", "--output_dir"):
            output_dir = arg

    f = open(txtfile, "r")
    dist = []
    for line in f:
        dist.append(line.split())
    dim = len(dist[0])+1
    edm = np.zeros((dim,dim))
    for i in range(0,dim-1):
        for j in range(i+1,10):
            edm[i,j] = float(dist[i][j-i-1])**2
    edm = edm + edm.T
    f.close()
    output_file = output_dir+'edm.txt'
    f = open(output_file, "w")
    for i in range(dim):
        for j in range(dim):
            f.write(str(edm[i,j])+' ')
        f.write("\n")
    f.close()

if __name__ == "__main__":
    create_edm(sys.argv[1:])