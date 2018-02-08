#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#define MaxReps 15
#define e 0.0001

//criterion for convergence
double frobeniusNorm(double **y,double **x,int NumberOfPoints,int Dimensions)
{
    int i,j;
    double sum=0;
    
    for (i=0;i<NumberOfPoints;i++)
    {
        for (j=0;j<Dimensions;j++)
        {
            sum = sum + (y[i][j]-x[i][j])*(y[i][j]-x[i][j]);
        }
    }
    
    return sqrt(sum);
}

//implementation of gaussian function
double gaussian(double x,double c)
{
    double f;
    f = exp(-x/(2*(c*c)));
        
    return f;
}

//make two arrays equal
void copy(double **y,double **x,int NumberOfPoints,int Dimensions)
{
    int i,j;
    for (i=0;i<NumberOfPoints ;i++)
    {
        for (j=0;j<Dimensions;j++)
        {
            y[i][j] = x[i][j];
        }
    }
}

//compute norm2 of a vector
double euclideanDistanceSquare(int i,int j,double **y,double **x,int Dimensions)
{
    double sum = 0;
    int l;
    
    for (l=0;l<Dimensions;l++)
    {
        sum = sum + pow((y[i][l]-x[j][l]),2);
    }
    
    return sum;
}

//mean shift algorithm
double meanShift(double **y,double **x,double c,int NumberOfPoints,int Dimensions)
{
    int i,j,k,g;
    double temp,sum,check;
    
    double **newY;
    newY = malloc(sizeof(double*) * NumberOfPoints);
    
    for (i=0;i<NumberOfPoints;i++)
    {
        newY[i] = malloc(sizeof(double) * Dimensions);
    }
    
    for (i=0;i<NumberOfPoints;i++)
        for (j=0;j<Dimensions;j++)
            newY[i][j] = 0;
    
    
    for (i=0;i<NumberOfPoints;i++)
    {
        sum = 0;
        
        for (j=0;j<NumberOfPoints;j++)
        {
            
            temp = euclideanDistanceSquare(i,j,y,x,Dimensions);
            
            if ((sqrt(temp)) > (c*c))
            {
                continue;
            }
            
            temp = gaussian(temp,c);
            
            //multiply all elements with temp
            for (k=0;k<Dimensions;k++)
            {
                newY[i][k] = newY[i][k] + temp * x[j][k];
            }
            
            sum = sum + temp;    
        }
        
        //devide all elements with sum
        for (k=0;k<Dimensions;k++)
        {
            newY[i][k] = newY[i][k] / sum;
        }
        
    }
    
    check = frobeniusNorm(newY,y,NumberOfPoints,Dimensions);
    
    for (i=0;i<NumberOfPoints;i++)
    {
        for (k=0;k<Dimensions;k++)
        {
            y[i][k] = newY[i][k];
        }
    }
    
    return check;
    
}
//test function
void test(double **y,int NumberOfPoints,int Dimensions,char *testFile)
{
    int i,j,n;
    double temp;
    double **yTest;
    
    yTest = malloc(sizeof(double*)*NumberOfPoints);
    
    for (i=0;i<NumberOfPoints;i++)
    {
        yTest[i] = malloc(sizeof(double)*Dimensions);
    }

    FILE *file = fopen(testFile, "r"); //open for read txt
   
    for (i=0;i<NumberOfPoints;i++)
    {
        for (j=0;j<Dimensions;j++)
        {
            n = fscanf(file,"%lf",&temp);
            yTest[i][j] = temp ;
        }
    }

    //compare the arrays
    for (i=0;i<NumberOfPoints;i++)
    {
        for (j=0;j<Dimensions;j++)
        {
            if (fabs(y[i][j]-yTest[i][j]) < 0.1)
            {
                continue;
            }
            else
            {
                
                printf("i:%d \n",i);
                printf("j:%d \n",j);
                printf("%f \n",yTest[i][j]);
                printf("%f \n",y[i][j]);
                
                printf("Test Failed!!! \n");   
                return;         
            }
        }
    }
    
    printf("Test Passed!!!! \n");
    
    fclose(file);
}

//load dataset drom a txt file
void loadData(double **x,int NumberOfPoints,int Dimensions,char *dataFile)
{
    
    int i,j,n; 
    
    FILE *file = fopen(dataFile, "r"); //open for read txt file
    
    for (i=0;i<NumberOfPoints;i++)
    {
        for (j=0;j<Dimensions;j++)
        {
           n = fscanf(file,"%lf",&x[i][j]);
        }
    }
    
    fclose(file);// close file
}


int main(int argc, char **argv)
{
    
    if (argc != 6) 
    {
        printf("Usage: %s <σ> <NumberOfPoints> <Dimensions> <DatasetFile> <TestFile> \n", argv[0]);
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
    
    int i,j,iterations=0;
    
    struct timeval startwtime, endwtime;
    double time,check;
    
    double **x;
    double **y;
    
    x = malloc(sizeof(double*)*NumberOfPoints);
    for (i=0;i<NumberOfPoints;i++)
    {
        x[i] = malloc(sizeof(double)*Dimensions);
    }
    
    y = malloc(sizeof(double*)*NumberOfPoints);
    for (i=0;i<NumberOfPoints;i++)
    {
        y[i] = malloc(sizeof(double)*Dimensions);
    }
    
    //load dataset
    loadData(x,NumberOfPoints,Dimensions,dataFile);
    
    //set y=x for the first iteration
    copy(y,x,NumberOfPoints,Dimensions);

    gettimeofday (&startwtime, NULL);
    
    do
    {
        iterations++;
        check = meanShift(y,x,c,NumberOfPoints,Dimensions);
        
        printf("error: %.8f \n",check);
        
    }while( (check>e) && (iterations<MaxReps)); 
      
    gettimeofday (&endwtime, NULL);  
    
    time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
		      + endwtime.tv_sec - startwtime.tv_sec);
              
    printf("serial mean shift time: %f \n", time);
    printf("iterations: %d \n",iterations);
    printf("%f \n",y[0][0]);
    printf("%f \n",y[0][1]);
    
    //test the results
    test(y,NumberOfPoints,Dimensions,testFile);
    
    return 0;
    
}
