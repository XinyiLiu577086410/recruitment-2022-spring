#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <cmath>
#include <chrono>
#include <vector>
#include <cassert>
#include <immintrin.h>
#include <omp.h>
#include <math.h>
#define PRINT_TIME(code) do { \
    auto start = system_clock::now(); \
    code \
    auto end   = system_clock::now(); \
    auto duration = duration_cast<microseconds>(end - start); \
    cout << "time spent: " << double(duration.count()) << "us" << endl; \
} while(0)

using namespace std;

using namespace chrono;

using vec = vector<int>; 

const int scale[] = {256, 512, 1024, 2048};
const string data_path("./data/");



//通用矩阵乘
//修改前的函数


// void Gemm(const int &size, vec &a, vec &b, vec &c) {
//     for(int i = 0; i < size; i++)
//         for(int j = 0; j < size; j++)
//             for(int k = 0; k < size; k++)
//                 c[i*size+j] += a[i*size+k] * b[k*size+j];
// }


/*version 1: cache blocking*/
// void Gemm(const int &size, vec &a, vec &b, vec &c) {
// #pragma omp parallel for schedule(dynamic,2) num_threads(128) shared(a,b,c,size)
//     for(int i = 0; i < size; i++)
//         for(int j = 0; j < size; j+=8)
//             for(int k = 0; k < size; k++)
//             {
//                 c[i*size+j] += a[i*size+k] * b[k*size+j];
//                 c[i*size+j+1] += a[i*size+k] * b[k*size+j+1];
//                 c[i*size+j+2] += a[i*size+k] * b[k*size+j+2];
//                 c[i*size+j+3] += a[i*size+k] * b[k*size+j+3];
//                 c[i*size+j+4] += a[i*size+k] * b[k*size+j+4];
//                 c[i*size+j+5] += a[i*size+k] * b[k*size+j+5];
//                 c[i*size+j+6] += a[i*size+k] * b[k*size+j+6];
//                 c[i*size+j+7] += a[i*size+k] * b[k*size+j+7];
//             }
// }


// /*version2:Transpose*/
// void Gemm(const int &size, vec &a, vec &b, vec &c) {
//     vec bTranspose(size*size,0);
//     for(int i = 0; i < size; i++)
//         for(int j = 0; j < size; j++){
//             bTranspose[j*size+i] = b[i*size+j];
//         }
//     #pragma omp parallel for schedule(dynamic,2) num_threads(128) shared(a,b,c,size)
//     for(int i = 0; i < size; i++)
//         for(int j = 0; j < size; j++)
//             for(int k = 0; k < size; k++)
//             {
//                 c[i*size+j] += a[i*size+k] * bTranspose[j*size+k];
//             }
// }

/*version2+:Transpose*/
// void Gemm(const int &size, vec &a, vec &b, vec &c) {
//     vec bTranspose(size*size,0);
//     #pragma omp parallel for schedule(dynamic,4) num_threads(128) shared(b,size)
//     for(int i = 0; i < size; i++)
//         for(int j = 0; j < size; j++){
//             bTranspose[j*size+i] = b[i*size+j];
//         }
//     #pragma omp parallel for schedule(dynamic,2) num_threads(128) shared(a,b,c,size)
//     for(int i = 0; i < size; i++)
//         for(int j = 0; j < size; j++)
//             for(int k = 0; k < size; k++)
//             {
//                 c[i*size+j] += a[i*size+k] * bTranspose[j*size+k];
//             }
// }

// /*version2.1+:Transpose,simd(avx512)*/
// void Gemm(const int &size, vec &a, vec &b, vec &c) {
//     vec bTranspose(size*size,0);
//     #pragma omp parallel for schedule(dynamic,4) num_threads(128) shared(b,size)
//     for(int i = 0; i < size; i++)
//         for(int j = 0; j < size; j++){
//             bTranspose[j*size+i] = b[i*size+j];
//         }
//     #pragma omp parallel for schedule(dynamic,2) num_threads(128) shared(a,b,c,size)
//     for(int i = 0; i < size; i++)
//         for(int j = 0; j < size; j++){
//             __m512i _a,_b,res,sum;
//             sum = _mm512_set_epi32(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
//             for(int k = 0; k < size; k += 16)
//             {
//                 _a = _mm512_loadu_si512(&a[0]+i*size+k);
//                 _b = _mm512_loadu_si512(&bTranspose[0]+j*size+k);
//                 res = _mm512_mullo_epi32(_a,_b);
//                 sum = _mm512_add_epi32(sum,res);
//             }
//             c[i*size+j] = _mm512_reduce_add_epi32(sum);
//         }
// }

// /*version2.2+:Transpose,simd(avx512)*/
// void Gemm(const int &size, vec &a, vec &b, vec &c) {
//     vec bTranspose(size*size,0);
//     int exp = log(size) / log(2);   //解决加速比倒挂的问题。
//     int numthr[] = {16, 32, 128, 128};
//     #pragma omp parallel for schedule(dynamic,4) num_threads(numthr[exp-8]) shared(b,size)
//     for(int i = 0; i < size; i++)
//         for(int j = 0; j < size; j++){
//             bTranspose[j*size+i] = b[i*size+j];
//         }
//     #pragma omp parallel for schedule(dynamic,4) num_threads(numthr[exp-8]) shared(a,b,c,size)
//     for(int i = 0; i < size; i++)
//         for(int j = 0; j < size; j++){
//             __m512i _a,_b,res,sum;
//             sum = _mm512_setzero_epi32();
//             for(int k = 0; k < size; k += 16)
//             {
//                 _a = _mm512_loadu_si512(&a[0]+i*size+k);
//                 _b = _mm512_loadu_si512(&bTranspose[0]+j*size+k);
//                 res = _mm512_mullo_epi32(_a,_b);
//                 sum = _mm512_add_epi32(sum,res);
//             }
//             c[i*size+j] = _mm512_reduce_add_epi32(sum);
//         }
// }
/*Version 2.4 load _a vector register outside the loop of k*/
void Gemm(const int &size, vec &a, vec &b, vec &c) {
    vec bTranspose(size*size,0);
    int exp = log(size) / log(2);   //解决加速比倒挂的问题。
    int numthr[] = {16, 32, 148, 148};
    #pragma omp parallel for schedule(dynamic,4) num_threads(numthr[exp-8]) shared(b,size)
    for(int i = 0; i < size; i++)
        for(int j = 0; j < size; j++){
            bTranspose[j*size+i] = b[i*size+j];
        }
    #pragma omp parallel for schedule(dynamic,4) num_threads(numthr[exp-8]) shared(a,b,c,size)
    for(int i = 0; i < size; i++)
        for(int k = 0; k < size; k += 16){
            __m512i _a = _mm512_loadu_si512(&a[0]+i*size+k);
            for(int j = 0; j < size; j++){
                // __m512i sum = _mm512_setzero_epi32();
                // sum = _mm512_add_epi32(sum,_mm512_mullo_epi32(_a,_mm512_loadu_si512(&bTranspose[0]+j*size+k)));
                c[i*size+j] += _mm512_reduce_add_epi32(
                                _mm512_mullo_epi32(_a,_mm512_loadu_si512(&bTranspose[0]+j*size+k))
                                );
            }
        }
}


/*Version 0:simd,no transpose*/
// void Gemm(const int &size, vec &a, vec &b, vec &c) {
    
//     __m256i __res, __b, __a,__sum,__sum1;
//     for(int i = 0; i < size; i++){
//         for(int j = 0; j < size; j += 8){
//             for(int k = 0; k < size; k++){
//                 int tmp[8];
//                 int aa = a[i*size+k];
//                 for (int i = 0; i < 8; i++) tmp[i] = aa;// 造向量。
//                 __a = _mm256_loadu_si256((__m256i const *)tmp);
//                 __b = _mm256_loadu_si256((__m256i const *)&b[0]+k*size+j);
//                 __res = _mm256_mullo_epi16(__a, __b);
//                 __sum1 = _mm256_add_epi32(__res,__sum);
//                 __sum = __sum1;
//             }
//             _mm256_storeu_si256((__m256i *)&c[0]+i*size+j, __sum);// 为什么要限定内存指针类型，莫名其妙的函数定义。
//         }
//     }

// }
/*
an error: ‘_mm512_storeu_epi32’ was not declared in this scope; did you mean ‘_mm512_store_epi32’?
*/

void CheckResult(const vec &c, const string &result_path) {
    ifstream file_result(result_path);
    int nelems = c.size();
    float res_i;
    for(int i = 0; i < nelems; i++) {
        file_result >> res_i;
        // std::cout<<i<<std::endl;
        assert(c[i] == res_i);
    }
    file_result.close();
}

// c = a * b
void Benchmark(const int &size) {
    const int nelems = size * size;
    const string a_path(data_path+to_string(size)+"/a");
    const string b_path(data_path+to_string(size)+"/b");
    const string result_path(data_path+to_string(size)+"/result");
    ifstream file_a(a_path);
    ifstream file_b(b_path);

    vec a(nelems, 0);
    vec b(nelems, 0);
    vec c(nelems, 0);

    for(int i = 0; i < nelems; i++) {
        file_a >> a[i];
    }
    for(int i = 0; i < nelems; i++) {
        file_b >> b[i];
    }

    PRINT_TIME(
       Gemm(size, a, b, c);
    );
    
    CheckResult(c, result_path);

    file_a.close();
    file_b.close();
}

int main() {
    for(auto size: scale) {
        cout << "Running, dataset: size " << size << endl;
        Benchmark(size);
        cout << "Passed, dataset: size " << size << endl;
        cout << endl;
    }
    return 0;
}