#include <stdio.h>
#include <thread>
#include <iostream>
#include "CycleTimer.h"

typedef struct {
    float x0, x1;
    float y0, y1;
    unsigned int width;
    unsigned int height;
    int maxIterations;
    int* output;
    int threadId;
    int numThreads;
} WorkerArgs;


extern void mandelbrotSerial(
    float x0, float y0, float x1, float y1,
    int width, int height,
    int startRow, int numRows,
    int maxIterations,
    int output[]);


//
// workerThreadStart --
//
// Thread entrypoint.
float step[][8]={{3,1.4,0.8,0.8,0.8,0.8,1.4,3},{7.0f/64, 7.0f/64, 7.0f/64, 9.1f/64, 9.1f/64, 9.1f/64, 9.1f/64, 7.0f/64}},start[][8]={{0,3,4.4,5.2,6,6.8,7.6,9},{0.0f, 7.0f/64, 14.0f/64, 21.0f/64, 30.0f/64, 39.0f/64, 48.0f/64, 57.0f/64}};
extern int k;
void workerThreadStart(WorkerArgs * const args) {

    // TODO FOR CS149 STUDENTS: Implement the body of the worker
    // thread here. Each thread should make a call to mandelbrotSerial()
    // to compute a part of the output image.  For example, in a
    // program that uses two threads, thread 0 could compute the top
    // half of the image and thread 1 could compute the bottom half.
    
    //int n = args->height/args->numThreads;
    // float step[] = {4.5,2.5,2.5,2.5,2.5,2.5,2.5,4.5}, start[] = {0,4.5,7,9.5,12,14.5,17,19.5};
    // float n = args->height/24;
    /*
    for(int i=0;i<8;i++){
        b[i] = b[i-1]+a[i]
    }
    */
    unsigned int n[] = {args->height/12,args->height};
    // std::cout<<"thread:"<<args->threadId<<": from "<<(int)(n[1]*start[1][args->threadId])<<" to "<<(int)(n[1]*start[1][args->threadId])+(int)(n[1]*step[1][args->threadId])<<std::endl;
    // std::cout<<"n[1]=="<<n[1]<<std::endl;
    // std::cout<<"step[0][args->threadId]=="<<(step[0][args->threadId])<<std::endl;
    //std::cout<<k<<std::endl;
    double startTime = CycleTimer::currentSeconds();
    mandelbrotSerial(
        args->x0, args->y0, args->x1, args->y1,
        args->width, args->height,
        //n*args->threadId,n,
        n[k]*start[k][args->threadId],n[k]*step[k][args->threadId]+1,
        //(int)(n*start[args->threadId]), (int)(n*step[args->threadId]+1),
        args->maxIterations,
        args->output);
    double endTime = CycleTimer::currentSeconds();
    printf("Hello world from thread %d : %.3lf ms\n", args->threadId, 1000*(endTime - startTime));
    

}

//
// MandelbrotThread --
//
// Multi-threaded implementation of mandelbrot set image generation.
// Threads of execution are created by spawning std::threads.
void mandelbrotThread(
    int numThreads,
    float x0, float y0, float x1, float y1,
    int width, int height,
    int maxIterations, int output[])
{
    static constexpr int MAX_THREADS = 32;

    if (numThreads > MAX_THREADS)
    {
        fprintf(stderr, "Error: Max allowed threads is %d\n", MAX_THREADS);
        exit(1);
    }

    // Creates thread objects that do not yet represent a thread.
    std::thread workers[MAX_THREADS];
    WorkerArgs args[MAX_THREADS];

    for (int i=0; i<numThreads; i++) {
      
        // TODO FOR CS149 STUDENTS: You may or may not wish to modify
        // the per-thread arguments here.  The code below copies the
        // same arguments for each thread
        args[i].x0 = x0;
        args[i].y0 = y0;
        args[i].x1 = x1;
        args[i].y1 = y1;
        args[i].width = width;
        args[i].height = height;
        args[i].maxIterations = maxIterations;
        args[i].numThreads = numThreads;
        args[i].output = output;
      
        args[i].threadId = i;
    }

    // Spawn the worker threads.  Note that only numThreads-1 std::threads
    // are created and the main application thread is used as a worker
    // as well.
    for (int i=1; i<numThreads; i++) {
        workers[i] = std::thread(workerThreadStart, &args[i]);
    }
    
    workerThreadStart(&args[0]);

    // join worker threads
    for (int i=1; i<numThreads; i++) {
        workers[i].join();
    }
}

