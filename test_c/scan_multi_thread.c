#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>

#include <windows.h>

#define NUM_ELEMENTS ((1 << 30))  // 1 billion elements
#define MAX_RANDOM_VALUE 10
#define NUM_PROCESSES 8u

typedef struct ProcessData 
{
    const uint32_t* input;
    uint32_t* output;
    uint32_t  length;
} ProcessData;

ProcessData g_process_data[NUM_PROCESSES];
uint32_t g_process_sums[NUM_PROCESSES] = {0, };
uint32_t g_offsets[NUM_PROCESSES] = {0, };
uint32_t g_process_temp[NUM_PROCESSES * 4] = {0, };

SYNCHRONIZATION_BARRIER g_barrier;


static inline uint32_t next_power_of_two(uint32_t n) {
    if (n == 0) return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}

void prescan_process(const uint32_t* input, uint32_t* output, uint32_t length, uint32_t process_index, uint32_t last_offset)
{
    uint32_t padded_length = next_power_of_two(length);
    uint32_t index = process_index * 2;
    uint32_t offset = 1;
    if (index < length)
        g_process_temp[index] = input[index];
    
    if (index + 1 < length)
        g_process_temp[index + 1] = input[index + 1];

    // upsweep
    for (uint32_t d = padded_length >> 1; d > 0; d >>= 1)
    {
        EnterSynchronizationBarrier(&g_barrier, SYNCHRONIZATION_BARRIER_FLAGS_BLOCK_ONLY);

        if (process_index < d)
        {
            uint32_t ai = offset * (2 * process_index + 1) - 1;
            uint32_t bi = offset * (2 * process_index + 2) - 1;
            g_process_temp[bi] += g_process_temp[ai];
        }

        offset <<= 1;
    }

    if (process_index == 0)
    {
        g_process_temp[padded_length - 1] = 0; // clear the last element
    }

    // downsweep
    for (uint32_t d = 1; d < padded_length; d <<= 1)
    {
        offset >>= 1;
        EnterSynchronizationBarrier(&g_barrier, SYNCHRONIZATION_BARRIER_FLAGS_BLOCK_ONLY);

        if (process_index < d)
        {
            uint32_t ai = offset * (2 * process_index + 1) - 1;
            uint32_t bi = offset * (2 * process_index + 2) - 1;

            uint32_t t = g_process_temp[ai];
            g_process_temp[ai] = g_process_temp[bi];
            g_process_temp[bi] += t;
        }

    }

    EnterSynchronizationBarrier(&g_barrier, SYNCHRONIZATION_BARRIER_FLAGS_BLOCK_ONLY);
    if (index < length)
        output[index] = g_process_temp[index] + last_offset;

    if (index + 1 < length)
        output[index + 1] = g_process_temp[index + 1] + last_offset;
}

void prescan_local(const uint32_t* input, uint32_t* output, uint32_t length, uint32_t offset)
{
    if (input == NULL || output == NULL || length == 0) 
        return;

    output[0] = offset;
    for (uint32_t i = 1; i < length; i++) 
    {
        output[i] = output[i - 1] + input[i - 1];
    }
}

DWORD process(LPVOID arg)
{
    uint32_t process_index = *((uint32_t*)arg);
    ProcessData* data = &g_process_data[process_index];
    const uint32_t* input = data->input;
    uint32_t* output = data->output;
    uint32_t length = data->length;

    if (input == NULL || output == NULL || length == 0) 
        return 0;

    // compute process sum
    uint32_t local_sum = 0;
    for (uint32_t i = 0; i < length; i++) 
    {
        local_sum += input[i];
    }
    g_process_sums[process_index] = local_sum;

    // wait for all processes to compute their sums
    EnterSynchronizationBarrier(&g_barrier, SYNCHRONIZATION_BARRIER_FLAGS_BLOCK_ONLY);
    
    // prescan on process sums to get offsets
    // upsweep
   prescan_process(g_process_sums, g_offsets, NUM_PROCESSES, process_index, 0);

   // debug
#if 0
    if (process_index == 0)
    {
         printf("Process Sums:\n");
         for (uint32_t p = 0; p < NUM_PROCESSES; p++)
         {
              printf("%u ", g_process_sums[p]);
         }
         printf("\n");
    
         printf("Offsets:\n");
         for (uint32_t p = 0; p < NUM_PROCESSES; p++)
         {
              printf("%u ", g_offsets[p]);
         }
         printf("\n");
    }
#endif

    // wait for all processes to finish prescan on process sums
    EnterSynchronizationBarrier(&g_barrier, SYNCHRONIZATION_BARRIER_FLAGS_BLOCK_ONLY);

    
    // prescan on its own section with offset

    // NOTE: we can't do multithreaded prescan as we can't control the number of threads for this last step.
    // So I just replace prescan_process with prescan_local below.
    // If we are in CUDA environment, we can launch a kernel with fixed number of blocks and threads to do this.
    // Therefore the prescan_process for getting offsets would be a good example to study parallel prescan algorithm.
    // prescan_process(data->total_input, data->total_output, NUM_ELEMENTS, process_index, g_offsets[process_index]);
    prescan_local(input, output, length, g_offsets[process_index]);

    return 0;
}

void prescan_with_processor_sum(const uint32_t* input, uint32_t* output, uint32_t length)
{
    uint32_t process_count = NUM_PROCESSES;
    uint32_t section_size = (length + process_count - 1) / process_count;

    HANDLE threads[NUM_PROCESSES];
    uint32_t process_indices[NUM_PROCESSES] = {0, };

    // initialize barrier
    InitializeSynchronizationBarrier(&g_barrier, process_count, -1);

    for (uint32_t p = 0; p < process_count; p++)
    {
        uint32_t start = p * section_size;
        uint32_t end = (p + 1) * section_size;
        if (end > length) end = length;
        if (start >= length) 
            continue;

        g_process_data[p].input = input + start;
        g_process_data[p].output = output + start;
        g_process_data[p].length = end - start;
        process_indices[p] = p;

        threads[p] = CreateThread(
            NULL,
            0,
            process,
            (LPVOID)(process_indices + p),
            0,
            NULL
        );
    }

    // wait for all threads to finish
    WaitForMultipleObjects(process_count, threads, TRUE, INFINITE);
    
    // cleanup threads
    for (uint32_t p = 0; p < process_count; p++)
    {
        CloseHandle(threads[p]);
    }
    
    // destroy barrier
    DeleteSynchronizationBarrier(&g_barrier);
}

int main()
{
    int seed_value = 20251016;
    srand(seed_value);

    uint32_t* rand_numbers = NULL;
    uint32_t* answer_numbers = NULL;
    uint32_t* test_numbers = NULL;
    rand_numbers = (uint32_t*)malloc(NUM_ELEMENTS * sizeof(uint32_t));
    if (rand_numbers == NULL) 
    {
        fprintf(stderr, "Memory allocation failed\n");
        goto EXIT;
    }

    answer_numbers = (uint32_t*)malloc(NUM_ELEMENTS * sizeof(uint32_t));
    if (answer_numbers == NULL) 
    {
        fprintf(stderr, "Memory allocation failed\n");
        goto EXIT;
    }

    test_numbers = (uint32_t*)malloc(NUM_ELEMENTS * sizeof(uint32_t));
    if (test_numbers == NULL) 
    {
        fprintf(stderr, "Memory allocation failed\n");
        goto EXIT;
    }

    for (uint32_t i = 0; i < NUM_ELEMENTS; i++) {
        rand_numbers[i] = rand() % MAX_RANDOM_VALUE;
        test_numbers[i] = 0; // prefetch memories
    }

    answer_numbers[0] = 0;
    for (uint32_t i = 1; i < NUM_ELEMENTS; i++) {
        answer_numbers[i] = answer_numbers[i - 1] + rand_numbers[i - 1];
    }

#if 0
    printf("Input Numbers:\n");
    for (uint32_t i = 0; i < NUM_ELEMENTS; i++) {
        printf("%u ", rand_numbers[i]);
    }
    printf("\n");

    printf("Prefix Sum Result:\n");
    for (uint32_t i = 0; i < NUM_ELEMENTS; i++) {
        printf("%u ", answer_numbers[i]);
    }
    printf("\n");
#endif

    time_t start_time = clock();

    prescan_with_processor_sum(rand_numbers, test_numbers, NUM_ELEMENTS);

    time_t end_time = clock();

    double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("Elapsed time for prescan_with_processor_sum: %.6f seconds\n", elapsed_time);

    bool success = true;
    for (uint32_t i = 0; i < NUM_ELEMENTS; i++) 
    {
        if (test_numbers[i] != answer_numbers[i]) 
        {
            success = false;
            break;
        }
    }

    if (success) 
    {
        printf("Test passed: The prefix sum results match the expected output.\n");
    } 
    else 
    {
        printf("Test failed: The prefix sum results do not match the expected output.\n");
        printf("Computed Result:\n");
#if 0
        for (uint32_t i = 0; i < NUM_ELEMENTS; i++) 
        {
            printf("%u ", test_numbers[i]);
        }
#endif
        printf("\n");
    }

EXIT:
    if (test_numbers) 
        free(test_numbers);

    if (answer_numbers) 
        free(answer_numbers);

    if (rand_numbers) 
        free(rand_numbers);

    return 0;
}