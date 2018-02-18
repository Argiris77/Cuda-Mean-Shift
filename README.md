# Mean Shift 

This is a mean shift algorithm implementation in cuda

### Three versions of the mean shift

- serialMeanShift.c: serial mean shift algorithm  

- cudaMeanShift.cu: parallel mean shift implementation with global memory

- cudaMeanShiftSharedMemory.cu: parallel mean shift implementation with shared memory

### To compile and run the programs

- clone this repo

- extract the data.zip file in this directory

- compile the programs with the make command

- run the programs with the commands:
```
  $./serialMeanShift -h- -NumberOfPoints- -Dimensions- -DatasetFile- -TestFile- 
  $./cudaMeanShift -h- -NumberOfPoints- -Dimensions- -DatasetFile- -TestFile- -BlocksPerGrid- -ThreadsPerBlock- 
  $./cudaMeanShiftSharedMemory -h- -NumberOfPoints- -Dimensions- -DatasetFile- -TestFile- -BlocksPerGrid- -ThreadsPerBlock- 
```
### Suggested execution for the datasets 

1. dataset 600x2 
```
   $./serialMeanShift 1 600 2 data600x2.txt test600x2.txt 
   $./cudaMeanShift 1 600 2 data600x2.txt test600x2.txt 6 100 
   $./cudaMeanShiftSharedMemory 1 600 2 data600x2.txt test600x2.txt 6 100
```   
2. dataset 1024x32:
```
   $./serialMeanShift 50 1024  32 data1024x32.txt test1024x32.txt 
   $./cudaMeanShift 50 1024 32 data1024x32.txt test1024x32.txt 8 128 
   $./cudaMeanShiftSharedMemory 50 1024 32 data1024x32.txt test1024x32.txt 8 128 
```   
3. dataset 2048x4:
```
   $./serialMeanShift 50 2048 4 data2048x4.txt test2048x4.txt 
   $./cudaMeanShift 50 2048 4 data2048x4.txt test2048x4.txt 8 256 
   $./cudaMeanShiftSharedMemory 50 2048 4 data2048x4.txt test2048x4.txt 8 256 
```   
4. dataset2048x16:
```
   $./serialMeanShift 50 2048 16 data2048x16.txt test2048x16.txt 
   $./cudaMeanShift 50 2048 16 data2048x16.txt test2048x16.txt 8 256 
   $./cudaMeanShiftSharedMemory 50 2048 16 data2048x16.txt test2048x16.txt 8 256 
```
                    
