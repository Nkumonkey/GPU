#include"gauss.cuh"
const int N = 1024;
const int BLOCK_SIZE = 1024;

int main(int argc, char** argv){
    int size=0;
    for (int i = 1; i < argc - 1; i = i + 2) {
            //argv[i]
        if (strcmp(argv[i], "--size") == 0)
            size = atoi(argv[i + 1]);
    }
    if(size<=0)
    size=N;
    
    Gauss <float>gauss(size, BLOCK_SIZE);
    
    gauss.simple_lu();
    gauss.cuda_lu(56);
    
    return 0;
}