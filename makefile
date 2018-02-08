all: serialMeanShift cudaMeanShift cudaMeanShiftSharedMemory

serialMeanShift:serialMeanShift.c
	gcc serialMeanShift.c -lm -O3 -o serialMeanShift

cudaMeanShift:cudaMeanShift.cu
	nvcc cudaMeanShift.cu -o cudaMeanShift -lm
	
cudaMeanShiftSharedMemory:cudaMeanShiftSharedMemory.cu
	nvcc cudaMeanShiftSharedMemory.cu -o cudaMeanShiftSharedMemory -lm
	
clear:
	rm serialMeanShift cudaMeanShift cudaMeanShiftSharedMemory
	


