#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <time.h>

#define NUM_ELEMENTS (1 << 30)  // 1 billion elements
#define MAX_RANDOM_VALUE 10
#define NUM_PROCESSES 8u

void prescan(const uint32_t* input, uint32_t* output, uint32_t length, uint32_t offset) 
{
    if (input == NULL || output == NULL || length == 0) 
        return;

    output[0] = offset;
    for (uint32_t i = 1; i < length; i++) 
    {
        output[i] = output[i - 1] + input[i - 1];
    }
}

void prescan_with_processor_sum(const uint32_t* input, uint32_t* output, uint32_t length) 
{
    if (input == NULL || output == NULL || length == 0) 
        return;

    uint32_t process_count = NUM_PROCESSES;
    uint32_t section_size = (length + process_count - 1) / process_count;
    
    // length 2 and process_count 4, then section_size = 1
    // the process 3 and 4 will not do anything
    
    // do processor sum first
    uint32_t process_sums[NUM_PROCESSES] = {0, };
    for (uint32_t p = 0; p < process_count; p++)
    {
        uint32_t start = p * section_size;
        uint32_t end = (p + 1) * section_size;
        if (end > length) end = length;
        if (start >= length) 
            continue;

        for (uint32_t i = start; i < end; i++) 
        {
            process_sums[p] += input[i];
        }
    }
    
    // prescan on processor sum
    uint32_t offsets[NUM_PROCESSES] = {0, };
    prescan(process_sums, offsets, process_count, 0);

    // prescan on each section
    for (uint32_t p = 0; p < process_count; p++)
    {
        uint32_t start = p * section_size;
        uint32_t end = (p + 1) * section_size;
        if (end > length) end = length;
        if (start >= length) 
            continue;
        
        uint32_t offset = offsets[p];
        prescan(input + start, output + start, end - start, offset);
    }
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

    /*
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
    */

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
        /*
        for (uint32_t i = 0; i < NUM_ELEMENTS; i++) 
        {
            printf("%u ", test_numbers[i]);
        }
        */
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