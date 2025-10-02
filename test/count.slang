StructuredBuffer<uint> keys : register(t0);
RWStructuredBuffer<uint> global_counts : register(u0);

#define NUM_KEYS 512
#define NUM_BINS 256
#define THREADS_PER_GROUP 16

// Shared memory: one count table per thread
groupshared uint local_count[THREADS_PER_GROUP][NUM_BINS];

[numthreads(THREADS_PER_GROUP, 1, 1)]
void main(uint3 global_id: SV_DispatchThreadID,
          uint3 local_id: SV_GroupThreadID,
          uint3 workgroup_id: SV_GroupID)
{
    uint tid = local_id.x;
    uint gid = global_id.x;

    // Initialize local count table
    for (uint i = 0; i < NUM_BINS; ++i)
    {
        local_count[tid][i] = 0;
    }
    GroupMemoryBarrierWithGroupSync();

    // Each thread processes multiple keys
    for (uint i = gid; i < NUM_KEYS; i += THREADS_PER_GROUP)
    {
        uint key = keys[i];
        if (key < NUM_BINS)
        {
            local_count[tid][key] += 1;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // Reduction - thread 0 sums all local tables
    if (tid == 0)
    {
        for (uint bin = 0; bin < NUM_BINS; ++bin)
        {
            uint sum = 0;
            for (uint t = 0; t < THREADS_PER_GROUP; ++t)
            {
                sum += local_count[t][bin];
            }
            InterlockedAdd(global_counts[bin], sum);
        }
    }
}