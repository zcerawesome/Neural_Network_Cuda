nvcc -Xcompiler -fPIC -Iinclude -c matrice_helper.cu -o matrice_helper.o
nvcc -shared -o lib/matrice_helper.so matrice_helper.o
