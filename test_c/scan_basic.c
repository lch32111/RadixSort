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

    prescan(rand_numbers, test_numbers, NUM_ELEMENTS, 0);

    time_t end_time = clock();

    double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("Elapsed time for prescan: %.6f seconds\n", elapsed_time);

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