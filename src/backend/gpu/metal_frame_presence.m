#import <arm_neon.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

typedef struct {
    const float *weights;
    const float *bias;
    size_t inputChannels, outputChannels, kernel, stride, dilation;
} FrameLayer;

API_AVAILABLE(macos(12.3))
@interface SubfastFrameBatch : NSObject
@property(nonatomic, strong) MPSGraphExecutable *executable;
@property(nonatomic, strong) NSArray<MPSGraphTensorData *> *inputs;
@property(nonatomic, strong) NSArray<MPSGraphTensorData *> *outputs;
@end

@implementation SubfastFrameBatch
@end

API_AVAILABLE(macos(12.3))
@interface SubfastFrameProgram : NSObject
@property(nonatomic, strong) MPSGraph *graph;
@property(nonatomic, strong) MPSGraphTensor *input;
@property(nonatomic, strong) MPSGraphTensor *output;
@end

@implementation SubfastFrameProgram
@end

API_AVAILABLE(macos(12.3))
@interface SubfastFrameLane : NSObject
@property(nonatomic, strong) id<MTLCommandQueue> queue;
@property(nonatomic, strong) id<MTLSharedEvent> completionEvent;
@property(nonatomic) uint64_t completionValue;
@property(nonatomic, strong) id<MTLBuffer> source;
@property(nonatomic, strong) id<MTLBuffer> scores;
@property(nonatomic, strong) NSMutableDictionary<NSNumber *, SubfastFrameBatch *> *batches;
@end

@implementation SubfastFrameLane
@end

API_AVAILABLE(macos(12.3))
@interface SubfastFrameGraph : NSObject
@property(nonatomic, strong) SubfastFrameProgram *nchw;
@property(nonatomic, strong) SubfastFrameProgram *nhwc;
@property(nonatomic, strong) MPSGraphDevice *graphDevice;
@property(nonatomic, strong) SubfastFrameLane *mainLane;
@property(nonatomic, strong) SubfastFrameLane *prefixLane;
@property(nonatomic) size_t height, width, maxBatch;
@end

@implementation SubfastFrameGraph
@end

static MPSGraphTensor *convolve(MPSGraph *graph, MPSGraphTensor *source, FrameLayer layer, BOOL relu) {
    NSUInteger k = layer.kernel;
    NSUInteger padding = (k / 2) * layer.dilation;
    MPSGraphTensor *weights = [graph
        constantWithData:[NSData dataWithBytes:layer.weights
            length:layer.outputChannels * layer.inputChannels * k * k * sizeof(float)]
        shape:@[@(k), @(k), @(layer.inputChannels), @(layer.outputChannels)]
        dataType:MPSDataTypeFloat32];
    MPSGraphTensor *bias = [graph
        constantWithData:[NSData dataWithBytes:layer.bias length:layer.outputChannels * sizeof(float)]
        shape:@[@1, @1, @1, @(layer.outputChannels)] dataType:MPSDataTypeFloat32];
    MPSGraphConvolution2DOpDescriptor *descriptor = [MPSGraphConvolution2DOpDescriptor
        descriptorWithStrideInX:layer.stride strideInY:layer.stride
        dilationRateInX:layer.dilation dilationRateInY:layer.dilation groups:1
        paddingStyle:MPSGraphPaddingStyleExplicit dataLayout:MPSGraphTensorNamedDataLayoutNHWC
        weightsLayout:MPSGraphTensorNamedDataLayoutHWIO];
    [descriptor setExplicitPaddingWithPaddingLeft:padding paddingRight:padding
        paddingTop:padding paddingBottom:padding];
    MPSGraphTensor *result = [graph convolution2DWithSourceTensor:source
        weightsTensor:weights descriptor:descriptor name:nil];
    result = [graph additionWithPrimaryTensor:result secondaryTensor:bias name:nil];
    return relu ? [graph reLUWithTensor:result name:nil] : result;
}

// Mid-sized batches benefit from combining the input copy and layout
// conversion on the host. Small batches avoid thread scheduling, and large
// batches use the bulk copy with the layout conversion on the GPU.
static const size_t interleavedMinBytes = 10 * 1024 * 1024;
static const size_t interleavedMaxBytes = 28 * 1024 * 1024;

static MPSShape *frameShape(NSNumber *batch, size_t height, size_t width, bool interleaved) {
    return interleaved ? @[batch, @(height), @(width), @3] : @[batch, @3, @(height), @(width)];
}

API_AVAILABLE(macos(12.3))
static SubfastFrameProgram *buildProgram(const FrameLayer layers[7], size_t height, size_t width,
    float scale, float bias, bool interleaved) {
    SubfastFrameProgram *program = [SubfastFrameProgram new];
    MPSGraph *graph = [MPSGraph new];
    program.graph = graph;
    program.input = [graph placeholderWithShape:frameShape(@(-1), height, width, interleaved)
        dataType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor *x = interleaved ? program.input :
        [graph transposeTensor:program.input permutation:@[@0, @2, @3, @1] name:nil];
    for (size_t i = 0; i < 3; ++i) {
        x = convolve(graph, x, layers[i], YES);
    }
    MPSGraphTensor *region = x;
    for (size_t i = 3; i < 5; ++i) {
        x = convolve(graph, x, layers[i], YES);
    }
    x = convolve(graph, x, layers[5], NO);
    x = [graph resizeTensor:x size:@[@(height / 8), @(width / 8)] mode:MPSGraphResizeBilinear
        centerResult:YES alignCorners:NO layout:MPSGraphTensorNamedDataLayoutNHWC name:nil];
    region = convolve(graph, region, layers[6], NO);
    x = [graph additionWithPrimaryTensor:region secondaryTensor:x name:nil];
    MPSGraphPooling2DOpDescriptor *pool = [MPSGraphPooling2DOpDescriptor new];
    pool.kernelWidth = pool.kernelHeight = 5;
    pool.strideInX = pool.strideInY = 1;
    pool.dilationRateInX = pool.dilationRateInY = 1;
    pool.paddingLeft = pool.paddingRight = pool.paddingTop = pool.paddingBottom = 2;
    pool.dataLayout = MPSGraphTensorNamedDataLayoutNHWC;
    pool.includeZeroPadToAverage = NO;
    x = [graph avgPooling2DWithSourceTensor:x descriptor:pool name:nil];
    x = [graph reductionMaximumWithTensor:x axes:@[@1, @2, @3] name:nil];
    x = [graph maximumWithPrimaryTensor:x
        secondaryTensor:[graph constantWithScalar:-10000 dataType:MPSDataTypeFloat32] name:nil];
    x = [graph multiplicationWithPrimaryTensor:x
        secondaryTensor:[graph constantWithScalar:scale dataType:MPSDataTypeFloat32] name:nil];
    x = [graph additionWithPrimaryTensor:x
        secondaryTensor:[graph constantWithScalar:bias dataType:MPSDataTypeFloat32] name:nil];
    program.output = [graph reshapeTensor:x withShape:@[@(-1)] name:nil];
    return program;
}

API_AVAILABLE(macos(12.3))
static SubfastFrameLane *makeLane(id<MTLDevice> device, size_t inputBytes, size_t maxBatch) {
    SubfastFrameLane *lane = [SubfastFrameLane new];
    lane.queue = [device newCommandQueue];
    if (@available(macOS 13.0, *)) {
        lane.completionEvent = [device newSharedEvent];
    }
    lane.source = [device newBufferWithLength:inputBytes options:MTLResourceStorageModeShared];
    lane.scores = [device newBufferWithLength:maxBatch * sizeof(float) options:MTLResourceStorageModeShared];
    lane.batches = [NSMutableDictionary new];
    return lane.queue && lane.source && lane.scores ? lane : nil;
}

void *subfast_frame_graph_create(const FrameLayer layers[7], size_t height, size_t width,
    size_t maxBatch, size_t inputBytes, float scale, float bias, char *name, size_t nameBytes) {
    if (@available(macOS 12.3, *)) {
        @autoreleasepool {
            @try {
                id<MTLDevice> device = MTLCreateSystemDefaultDevice();
                if (device == nil || inputBytes > device.maxBufferLength) {
                    return NULL;
                }
                SubfastFrameGraph *state = [SubfastFrameGraph new];
                state.mainLane = makeLane(device, inputBytes, maxBatch);
                if (state.mainLane == nil) {
                    return NULL;
                }
                state.graphDevice = [MPSGraphDevice deviceWithMTLDevice:device];
                state.height = height;
                state.width = width;
                state.maxBatch = maxBatch;
                state.nchw = buildProgram(layers, height, width, scale, bias, false);
                size_t frameBytes = height * width * 3 * sizeof(float);
                if (inputBytes >= interleavedMinBytes && frameBytes <= interleavedMaxBytes) {
                    state.nhwc = buildProgram(layers, height, width, scale, bias, true);
                }
                snprintf(name, nameBytes, "%s", device.name.UTF8String);
                return (__bridge_retained void *)state;
            } @catch (NSException *exception) {
                return NULL;
            }
        }
    }
    return NULL;
}

API_AVAILABLE(macos(12.3))
static SubfastFrameBatch *compileBatch(SubfastFrameGraph *state, SubfastFrameLane *lane, size_t batch, bool interleaved) {
    SubfastFrameProgram *program = interleaved ? state.nhwc : state.nchw;
    MPSShape *shape = frameShape(@(batch), state.height, state.width, interleaved);
    MPSGraphShapedType *type = [[MPSGraphShapedType alloc] initWithShape:shape dataType:MPSDataTypeFloat32];
    MPSGraphCompilationDescriptor *options = [MPSGraphCompilationDescriptor new];
    // Level 0 keeps graph placement on the GPU.
    options.optimizationLevel = MPSGraphOptimizationLevel0;
    SubfastFrameBatch *plan = [SubfastFrameBatch new];
    plan.executable = [program.graph compileWithDevice:state.graphDevice feeds:@{program.input: type}
        targetTensors:@[program.output] targetOperations:nil compilationDescriptor:options];
    MPSGraphTensorData *input = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:lane.source shape:shape dataType:MPSDataTypeFloat32];
    MPSGraphTensorData *output = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:lane.scores shape:@[@(batch)] dataType:MPSDataTypeFloat32];
    if (plan.executable == nil || input == nil || output == nil) {
        return nil;
    }
    plan.inputs = @[input];
    plan.outputs = @[output];
    return plan;
}

static void uploadInput(void *destination, const void *input, size_t bytes) {
    // A single core is enough for smaller batches. Parallel copying large
    // batches gets more of the shared-memory bandwidth before GPU submission.
    if (bytes < 32 * 1024 * 1024) {
        memcpy(destination, input, bytes);
        return;
    }
    dispatch_apply(4, DISPATCH_APPLY_AUTO, ^(size_t index) {
        size_t chunk = bytes / 4;
        size_t start = index * chunk;
        size_t end = index == 3 ? bytes : start + chunk;
        memcpy((char *)destination + start, (const char *)input + start, end - start);
    });
}

static void uploadInterleaved(void *destination, const void *input, size_t pixels, size_t batch) {
    // Each shuffle selects four f32 values from four consecutive planar pixels.
    // Three shuffles produce the twelve f32 values in RGBRGB... order.
    const uint8x16_t first = {0,1,2,3, 16,17,18,19, 32,33,34,35, 4,5,6,7};
    const uint8x16_t second = {20,21,22,23, 36,37,38,39, 8,9,10,11, 24,25,26,27};
    const uint8x16_t third = {40,41,42,43, 12,13,14,15, 28,29,30,31, 44,45,46,47};
    size_t lanes = MIN((size_t)6, batch);
    size_t planeBytes = pixels * sizeof(float);
    size_t frameBytes = planeBytes * 3;
    dispatch_apply(lanes, DISPATCH_APPLY_AUTO, ^(size_t lane) {
        size_t end = batch * (lane + 1) / lanes;
        for (size_t frame = batch * lane / lanes; frame < end; ++frame) {
            const uint8_t *red = (const uint8_t *)input + frame * frameBytes;
            const uint8_t *green = red + planeBytes;
            const uint8_t *blue = green + planeBytes;
            uint8_t *rgb = (uint8_t *)destination + frame * frameBytes;
            // Frame Presence dimensions are multiples of eight.
            for (size_t offset = 0; offset < planeBytes; offset += 16) {
                uint8x16x3_t planes = {{
                    vld1q_u8(red + offset), vld1q_u8(green + offset), vld1q_u8(blue + offset)
                }};
                vst1q_u8(rgb, vqtbl3q_u8(planes, first));
                vst1q_u8(rgb + 16, vqtbl3q_u8(planes, second));
                vst1q_u8(rgb + 32, vqtbl3q_u8(planes, third));
                rgb += 48;
            }
        }
    });
}

API_AVAILABLE(macos(12.3))
static SubfastFrameBatch *batchPlan(SubfastFrameGraph *state, SubfastFrameLane *lane,
    size_t batch, bool interleaved) {
    SubfastFrameBatch *plan = lane.batches[@(batch)];
    if (plan == nil) {
        plan = compileBatch(state, lane, batch, interleaved);
        if (plan == nil) {
            return nil;
        }
        // A video workload typically alternates a full batch and a tail.
        if (lane.batches.count >= 4) {
            [lane.batches removeObjectForKey:lane.batches.keyEnumerator.nextObject];
        }
        lane.batches[@(batch)] = plan;
    }
    return plan;
}

static bool usesInterleaved(size_t bytes) {
    return bytes >= interleavedMinBytes && bytes <= interleavedMaxBytes;
}

static void writeError(NSException *exception, char *error, size_t errorBytes) {
    const char *message = exception.reason.UTF8String;
    if (message != NULL) {
        snprintf(error, errorBytes, "%s", message);
    }
}

API_AVAILABLE(macos(13.0))
static bool executeWithEvent(SubfastFrameLane *lane, SubfastFrameBatch *plan, size_t batch) {
    id<MTLSharedEvent> event = lane.completionEvent;
    uint64_t value = lane.completionValue + 1;
    lane.completionValue = value;
    dispatch_semaphore_t completion = dispatch_semaphore_create(0);
    __block BOOL success = NO;
    MPSGraphExecutableExecutionDescriptor *descriptor = [MPSGraphExecutableExecutionDescriptor new];
    // The callback can outlive this invocation when the device event wins.
    // It owns its state and does not write to the caller's input or output.
    descriptor.completionHandler = ^(NSArray *results, NSError *error) {
        success = error == nil;
        dispatch_semaphore_signal(completion);
    };
    [descriptor signalEvent:event atExecutionEvent:MPSGraphExecutionStageCompleted value:value];
    NSArray *results = [plan.executable runAsyncWithMTLCommandQueue:lane.queue
        inputsArray:plan.inputs resultsArray:plan.outputs executionDescriptor:descriptor];
    if (results.count != 1) {
        return false;
    }

    if (batch <= 12) {
        // Short graphs can finish before the host completion callback is
        // scheduled. Bound the spin and keep a blocking path for slower calls.
        uint64_t deadline = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) + 1800000;
        do {
            for (size_t poll = 0; poll < 32; ++poll) {
                if (event.signaledValue >= value) {
                    return true;
                }
                if (dispatch_semaphore_wait(completion, DISPATCH_TIME_NOW) == 0) {
                    return success;
                }
            }
        } while (clock_gettime_nsec_np(CLOCK_UPTIME_RAW) < deadline);
    } else if ([event waitUntilSignaledValue:value timeoutMS:20]) {
        return true;
    }

    dispatch_semaphore_wait(completion, DISPATCH_TIME_FOREVER);
    return success;
}

API_AVAILABLE(macos(12.3))
static bool executePlan(SubfastFrameLane *lane, SubfastFrameBatch *plan, size_t batch) {
    if (@available(macOS 13.0, *)) {
        // The synchronous entry point has lower overhead for one frame.
        if (batch > 1 && lane.completionEvent && lane.completionValue != UINT64_MAX) {
            return executeWithEvent(lane, plan, batch);
        }
    }
    return [plan.executable runWithMTLCommandQueue:lane.queue inputsArray:plan.inputs
        resultsArray:plan.outputs executionDescriptor:nil].count == 1;
}

API_AVAILABLE(macos(12.3))
static bool runPlan(SubfastFrameLane *lane, SubfastFrameBatch *plan, const void *input,
    size_t bytes, size_t pixels, size_t batch, bool interleaved, float *output,
    char *error, size_t errorBytes) {
    @autoreleasepool {
        @try {
            if (interleaved) {
                uploadInterleaved(lane.source.contents, input, pixels, batch);
            } else {
                uploadInput(lane.source.contents, input, bytes);
            }
            if (!executePlan(lane, plan, batch)) {
                return false;
            }
            memcpy(output, lane.scores.contents, batch * sizeof(float));
            return true;
        } @catch (NSException *exception) {
            writeError(exception, error, errorBytes);
            return false;
        }
    }
}

API_AVAILABLE(macos(12.3))
static bool runOrderedHalves(SubfastFrameLane *prefix, SubfastFrameBatch *first,
    SubfastFrameLane *main, SubfastFrameBatch *second, const void *input,
    size_t pixels, float *output, char *error, size_t errorBytes) {
    const size_t count = 8;
    const size_t bytes = count * pixels * 3 * sizeof(float);
    uploadInterleaved(prefix.source.contents, input, pixels, count);
    dispatch_semaphore_t completion = dispatch_semaphore_create(0);
    __block BOOL firstOK = NO;
    MPSGraphExecutableExecutionDescriptor *descriptor = [MPSGraphExecutableExecutionDescriptor new];
    descriptor.completionHandler = ^(NSArray *results, NSError *error) {
        firstOK = error == nil;
        dispatch_semaphore_signal(completion);
    };
    NSArray *results = [first.executable runAsyncWithMTLCommandQueue:prefix.queue
        inputsArray:first.inputs resultsArray:first.outputs executionDescriptor:descriptor];
    if (results.count != 1) {
        return false;
    }

    // Submit the prefix before the second interleaved copy. Both copies can
    // use the host workers without competing for memory bandwidth.
    bool secondOK = runPlan(main, second, (const uint8_t *)input + bytes,
        bytes, pixels, count, true, output + count, error, errorBytes);
    dispatch_semaphore_wait(completion, DISPATCH_TIME_FOREVER);
    if (firstOK && secondOK) {
        memcpy(output, prefix.scores.contents, count * sizeof(float));
        return true;
    }
    return false;
}

bool subfast_frame_graph_run(void *handle, const void *input, size_t inputBytes, float *output,
    size_t batch, char *error, size_t errorBytes) {
    if (@available(macOS 12.3, *)) {
        @autoreleasepool {
            @try {
                SubfastFrameGraph *state = (__bridge SubfastFrameGraph *)handle;
                size_t pixels = state.height * state.width;
                if (batch == 0 || batch > state.maxBatch ||
                    inputBytes != batch * pixels * 3 * sizeof(float)) {
                    return false;
                }
                SubfastFrameLane *main = state.mainLane;
                // Split the measured video batch across two queues so that
                // host upload can overlap GPU work on the other part.
                if ((batch == 32 || (batch == 16 && main.completionEvent != nil)) &&
                    state.height == 272 && state.width == 480) {
                    const size_t prefixCount = batch == 32 ? 12 : 8;
                    const size_t prefixCapacity = state.maxBatch >= 32 ? 12 : 8;
                    size_t frameBytes = pixels * 3 * sizeof(float);
                    size_t prefixBytes = prefixCount * frameBytes;
                    size_t suffixCount = batch - prefixCount;
                    size_t suffixBytes = inputBytes - prefixBytes;
                    bool suffixInterleaved = usesInterleaved(suffixBytes);
                    if (state.prefixLane == nil) {
                        state.prefixLane = makeLane(main.queue.device, prefixCapacity * frameBytes, prefixCapacity);
                    }
                    SubfastFrameLane *prefix = state.prefixLane;
                    if (prefix != nil) {
                        SubfastFrameBatch *first = batchPlan(state, prefix, prefixCount, true);
                        SubfastFrameBatch *second = batchPlan(state, main, suffixCount, suffixInterleaved);
                        if (first != nil && second != nil) {
                            if (batch == 16) {
                                return runOrderedHalves(prefix, first, main, second, input, pixels,
                                    output, error, errorBytes);
                            }
                            __block bool firstOK = false, secondOK = false;
                            char firstError[512] = {0};
                            char *firstErrorPtr = firstError;
                            dispatch_apply(2, DISPATCH_APPLY_AUTO, ^(size_t task) {
                                if (task == 0) {
                                    firstOK = runPlan(prefix, first, input, prefixBytes, pixels,
                                        prefixCount, true, output, firstErrorPtr, sizeof firstError);
                                } else {
                                    secondOK = runPlan(main, second, (const uint8_t *)input + prefixBytes,
                                        suffixBytes, pixels, suffixCount, suffixInterleaved,
                                        output + prefixCount, error, errorBytes);
                                }
                            });
                            if (!firstOK && firstError[0] != '\0') {
                                snprintf(error, errorBytes, "%s", firstError);
                            }
                            return firstOK && secondOK;
                        }
                    }
                }
                bool interleaved = state.nhwc != nil && usesInterleaved(inputBytes);
                SubfastFrameBatch *plan = batchPlan(state, main, batch, interleaved);
                return plan != nil && runPlan(main, plan, input, inputBytes, pixels, batch,
                    interleaved, output, error, errorBytes);
            } @catch (NSException *exception) {
                writeError(exception, error, errorBytes);
            }
        }
    }
    return false;
}

void subfast_frame_graph_destroy(void *handle) {
    @autoreleasepool {
        CFBridgingRelease(handle);
    }
}
