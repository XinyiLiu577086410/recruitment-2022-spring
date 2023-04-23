gemm: main.cpp
	g++ main.cpp -o gemm -O2 -Wall -Werror -std=c++11 -mavx512f -fopenmp

.PHONY: clean
clean:
	- rm -f gemm

.PHONY: run
run: 
	./gemm

.PHONY: rmdata
rmdata:
	- rm -rf data*	 