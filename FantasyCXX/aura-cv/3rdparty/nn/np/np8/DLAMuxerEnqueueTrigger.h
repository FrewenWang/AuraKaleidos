#pragma once

#include "neuron/api/Fence.h"

__BEGIN_DECLS

// Introduction to Enqueue Trigger mechanism.
// Enqueue Trigger is an inference mechanism that can separate inference into two stage,
// (1) enqueue-stage and (2) trigger-stage.
// All the pre-execution tasks would be done in enqueue-stage. Then in trigger-stage we
// will only trigger the enqueued job to execute.
// Once the inference settings are changed. User should enqueue the job one more time.
// Then after that user can trigger that job any time they want before session released.

/**
 * Check if the model supports enqueThenTrigger execution. Call this function after
 * runtime is loaded with model.
 * @param runtime The address of the created neuron runtime instance.
 * @param supported Non-zero value indicates that the model supports enqueueThenTrigger
 * execution.
 * @return An error code indicates whether the test model executes successfully.
 */
int NeuronDLAMuxer_isEnqueueTriggerSupported(void* dlaMuxer, uint8_t* supported);

/**
 * Do job-enqueue. All the pre-process stuffs will be done at this stage, then the runtime job
 * will be enqueue into kernel-space to waiting for the trigger signal.
 * @param runtime The address of the created neuron runtime instance.
 * @return A Runtime error code.
 */
int NeuronDLAMuxer_inferenceEnqueue(void* dlaMuxer);

/**
 * Do job-trigger. Trigger job that user enqueued before. It is expected that
 * NeuronDLAMuxer_inferenceEnqueue() has been called before and it is the last API
 * invoked before calling this API.
 * @param runtime The address of the created neuron runtime instance.
 * @return A Runtime error code.
 */
int NeuronDLAMuxer_inferenceTrigger(void* dlaMuxer);

/**
 * Do job-trigger-with-fence. Trigger job that user enqueued before. It is expected that
 * NeuronDLAMuxer_inferenceEnqueue() has been called before and it is the last API
 * invoked before calling this API.
 * The call should return without waiting for inference to finish. The caller should
 * prepare a FenceInfo structure and pass its address into this API. FenceFd in FenceInfo
 * will be set and the caller can be signaled when inference completes (or error exit) by
 * waiting on the fence. Most importantly, after the fence is triggered, caller MUST call
 * the callback in fenceInfo so that Neuron can perform certain post-execution tasks. The
 * final execution status and inference time can be retrieved in FenceInfo after the
 * callback is executed.
 * @param runtime The address of the created neuron runtime instance.
 * @return A Runtime error code.
 */
int NeuronDLAMuxer_inferenceTriggerFenced(void* dlaMuxer, FenceInfo* fenceInfo);

__END_DECLS
