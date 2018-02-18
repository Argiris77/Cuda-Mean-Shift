#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define MaxReps 15
#define e 0.0001

//criterion for convergence
double frobeniusNorm(double *y,size_t pitchY,double *x,size_t pitchX,int NumberOfPoints,int Dimensions)
{
    int i,j;
    double sum=0;
    double *rowX,*rowY;
    
    for (i=0;i<NumberOfPoints;i++)
    {
        rowY = (double*)((char*)y + i * pitchY);
        rowX = (double*)((char*)x + i * pitchX);
        for (j=0;j<Dimensions;j++)
        {
            sum = sum + (rowY[j]-rowX[j])*(rowY[j]-rowX[j]);
        }
    }
    
    return sqrt(sum);
}

//implementation of gaussian function
__device__ double gaussian(double x,double c)
{
    double f;
    f = exp(-x/(2*(c*c)));
        
    return f;
}

//make two arrays equal
void copy(double *y,size_t pitchY,double *x,size_t pitchX,int NumberOfPoints,int Dimensions)
{
    int i,j;
    double *rowY,*rowX;
    for (i=0;i<NumberOfPoints ;i++)
    {
        rowY = (double*)((char*)y + i * pitchY);
        rowX = (double*)((char*)x +i * pitchX);
        for (j=0;j<Dimensions;j++)
        {
            rowY[j] = rowX[j];
        }
    }
}

//compute norm2 of a vector
__device__ double euclideanDistanceSquare(int index,int j,double *y,size_t pitch2,double *x,size_t pitch1,int Dimensions)
{
    double sum = 0;
    int l;
    double *row1 = (double*)((char*)y + index * pitch2);
    double *row2 = (double*)((char*)x + j * pitch1);
    
    for (l=0;l<Dimensions;l++)
    {
        sum = sum + pow((row1[l]-row2[l]),2);
    }
    
    return sum;
}

//mean shift algorithm(device code)
__global__ void meanShift(double *y,size_t pitch2,double *x,size_t pitch1,double *newY,size_t pitch3,int NumberOfPoints,int Dimensions,double c)
{
    
    int j,k;
    double temp,sum;
    double *row1;
    double *row2;
    double *row3;
   
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (index < NumberOfPoints)
    {
        
        row3 = (double*)((char*)newY + index * pitch3 ); 
        //row2 = (double*)((char*)y + index * pitch2);
        sum = 0;
        
        for (k=0;k<Dimensions;k++)
        {
            row3[k] = 0;
        }
        
        
        for (j=0;j<NumberOfPoints;j++)
        {
            temp = euclideanDistanceSquare(index,j,y,pitch2,x,pitch1,Dimensions);
            
            if ((sqrt(temp)) > (c*c))
            {
                continue;
            }
            
            temp = gaussian(temp,c);
            
            
            if (index==j)
                temp = 1;
            
            
            row1 = (double*)((char*)x + j * pitch1);
             
            for (k=0;k<Dimensions;k++)
            {
                row3[k] = row3[k] + temp * row1[k];
            }
            
            sum = sum + temp;    
        }
        
        for (k=0;k<Dimensions;k++)
        {
            row3[k] = row3[k] / sum;
        }
        
        //check = frobeniusNorm(newY,y);
        
        row2 = (double*)((char*)y + index * pitch2);
        
        for (k=0;k<Dimensions;k++)
        {
            row2[k] = row3[k];
        }
        
        
    }
    
    
    //return check;
    
}

//test function
void test(double **y,int NumberOfPoints,int Dimensions,char *testFile)
{
    int i,j;
    double temp;
    double **yTest;
    
    yTest =(double **) malloc(sizeof(double*)*NumberOfPoints);
    
    for (i=0;i<NumberOfPoints;i++)
    {
        yTest[i] = (double *)malloc(sizeof(double)*Dimensions);
    }

    FILE *file = fopen(testFile, "r"); //open for read txt
    //read the data from kknsearch function of matlab
    for (i=0;i<NumberOfPoints;i++)
    {
        for (j=0;j<Dimensions;j++)
        {
            fscanf(file,"%lf",&temp);
            yTest[i][j] = temp ;
        }
    }

    //compare the arrays
    for (i=0;i<NumberOfPoints;i++)
    {
        for (j=0;j<Dimensions;j++)
        {
            if (fabs(y[i][j]-yTest[i][j]) < 1)
            {
                continue;
            }
            else
            {
                printf("Test Failed!!! \n");   
                return;         
            }
        }
    }
    
    printf("Test Passed!!!! \n");
    
    fclose(file);
}


//load dataset drom a txt file
void loadData(double *x,size_t pitchX,int NumberOfPoints,int Dimensions,char *dataFile)
{
    
    int i,j; 
    double *row; 
    FILE *file = fopen(dataFile, "r"); //open for read txt file
    
    for (i=0;i<NumberOfPoints;i++)
    {
        row = (double*)((char*)x + i * pitchX );
        for (j=0;j<Dimensions;j++)
        {
        fscanf(file,"%lf",&row[j]);
        }
    }
    
    fclose(file);// close file
}


int main(int argc, char **argv)
{
    
    
    
    if (argc != 8) 
    {
        printf("Usage: %s <σ> <NumberOfPoints> <Dimensions> <DatasetFile> <TestFile> <BlocksPerGrid> <ThreadsPerBlock> \n", argv[0]);
        exit(1);
    }
    
    double c;
    c = atof(argv[1]);
    
    int NumberOfPoints = atoi(argv[2]);
    int Dimensions = atoi(argv[3]);
    
    char *dataFile;
    dataFile = argv[4];
    char *testFile;
    testFile = argv[5];
    
    int threadsPerBlock = atoi(argv[7]);
    int blocksPerGrid = atoi(argv[6]); 
    
    
    int iterations=0;
    struct timeval startwtime, endwtime;
    double time,error;
    
    //allocate arrays for host memory
    double *y;
    size_t pitch = sizeof(double) * Dimensions;
    y = (double*)malloc(sizeof(double) * NumberOfPoints * Dimensions);
    
    double *yNew;
    size_t pitchNew = sizeof(double) * Dimensions;
    yNew = (double*)malloc(sizeof(double) * NumberOfPoints * Dimensions);
    
    double *x;
    size_t pitchX = sizeof(double) * Dimensions;
    x = (double*)malloc(sizeof(double) * NumberOfPoints * Dimensions);
    
    //load dataset from txt file
    loadData(x,pitchX,NumberOfPoints,Dimensions,dataFile);
    
    //set y=x for the first iteration
    copy(y,pitch,x,pitchX,NumberOfPoints,Dimensions);
    
    //allocate 2d arrays for device memory
    double *d_x;
    double *d_y;
    double *d_yNew;
    size_t pitch1,pitch2,pitch3;
    
    cudaMallocPitch((void**)&d_x, &pitch1, Dimensions * sizeof(double), NumberOfPoints);
    cudaMemcpy2D(d_x,pitch1,x,Dimensions * sizeof(double), Dimensions * sizeof(double), NumberOfPoints, cudaMemcpyHostToDevice);
    cudaMallocPitch((void**)&d_y, &pitch2, Dimensions * sizeof(double), NumberOfPoints);
    cudaMemcpy2D(d_y,pitch2,y,Dimensions * sizeof(double), Dimensions * sizeof(double), NumberOfPoints, cudaMemcpyHostToDevice);
    cudaMallocPitch((void**)&d_yNew, &pitch3, Dimensions * sizeof(double), NumberOfPoints);
    
    gettimeofday (&startwtime, NULL);
    
    do
    {
        iterations++;
        copy(yNew,pitchNew,y,pitch,NumberOfPoints,Dimensions);
        meanShift<<<blocksPerGrid,threadsPerBlock>>>(d_y,pitch2,d_x,pitch1,d_yNew,pitch3,NumberOfPoints,Dimensions,c);
        
        cudaMemcpy2D(y, sizeof(double)*Dimensions, d_y, pitch2, sizeof(double) * Dimensions, NumberOfPoints, cudaMemcpyDeviceToHost);
        error = frobeniusNorm(y,pitch,yNew,pitchNew,NumberOfPoints,Dimensions);
        
    }while((error>e));
    
    gettimeofday (&endwtime, NULL); 
    
    
    
    int i,j,b = 0;
    
    double **yarray;
    yarray = (double **)malloc(sizeof(double *)*NumberOfPoints);
    for (i=0;i<NumberOfPoints;i++)
    {
        yarray[i] = (double *)malloc(sizeof(double)*Dimensions);
    }
    
    for(i=0;i<NumberOfPoints;i++)
    {
        for (j=0;j<Dimensions;j++)
        {
            yarray[i][j] = y[b];
            b++;
        }
    }
    
    time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
            + endwtime.tv_sec - startwtime.tv_sec);
            
    printf("cuda mean shift time: %f \n", time);
    printf("iterations: %d \n",iterations);
    
    //test the results
    test(yarray,NumberOfPoints,Dimensions,testFile);
   
    
    return 0;
    
}
