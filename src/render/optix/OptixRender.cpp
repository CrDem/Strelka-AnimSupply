#include "OptixRender.h"

#include "OptixBuffer.h"

#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>

#include <glm/glm.hpp>
#include <glm/mat4x3.hpp>
#include <glm/gtx/compatibility.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/matrix_major_storage.hpp>
#include <glm/ext/matrix_relational.hpp>

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <vector_types.h>
#include <vector_functions.h>

#include <sutil/vec_math_adv.h>

#include "texture_support_cuda.h"

#include <filesystem>
#include <array>
#include <string>
#include <fstream>
#include <memory>

#include <log.h>

#include "cuda_checks.h"
#include "postprocessing/Tonemappers.h"

#include "Camera.h"

static void context_log_cb(unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
    switch (level)
    {
    case 1:
        STRELKA_FATAL("OptiX [{0}]: {1}", tag, message);
        break;
    case 2:
        STRELKA_ERROR("OptiX [{0}]: {1}", tag, message);
        break;
    case 3:
        STRELKA_WARNING("OptiX [{0}]: {1}", tag, message);
        break;
    case 4:
        STRELKA_INFO("OptiX [{0}]: {1}", tag, message);
        break;
    default:
        break;
    }
}

static inline void optixCheck(OptixResult res, const char* call, const char* file, unsigned int line)
{
    if (res != OPTIX_SUCCESS)
    {
        STRELKA_ERROR("OptiX call {0} failed: {1}:{2}", call, file, line);
        assert(0);
    }
}

static inline void optixCheckLog(OptixResult res,
                                 const char* log,
                                 size_t sizeof_log,
                                 size_t sizeof_log_returned,
                                 const char* call,
                                 const char* file,
                                 unsigned int line)
{
    if (res != OPTIX_SUCCESS)
    {
        STRELKA_FATAL("OptiX call {0} failed: {1}:{2} : {3}", call, file, line, log);
        assert(0);
    }
}

//------------------------------------------------------------------------------
//
// OptiX error-checking
//
//------------------------------------------------------------------------------
#define OPTIX_CHECK(call) optixCheck(call, #call, __FILE__, __LINE__)
#define OPTIX_CHECK_LOG(call) optixCheckLog(call, log, sizeof(log), sizeof_log, #call, __FILE__, __LINE__)


using namespace oka;
namespace fs = std::filesystem;

template <typename T>
struct SbtRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RayGenData> RayGenSbtRecord;
typedef SbtRecord<MissData> MissSbtRecord;
typedef SbtRecord<HitGroupData> HitGroupSbtRecord;

static bool readSourceFile(std::string& str, const fs::path& filename)
{
    // Try to open file
    std::ifstream file(filename.c_str(), std::ios::binary);
    if (file.good())
    {
        // Found usable source file
        std::vector<unsigned char> buffer = std::vector<unsigned char>(std::istreambuf_iterator<char>(file), {});
        str.assign(buffer.begin(), buffer.end());
        return true;
    }
    return false;
}

OptiXRender::OptiXRender()
{
}

OptiXRender::~OptiXRender()
{
}

void OptiXRender::createContext()
{
    // Initialize CUDA
    CUDA_CHECK(cudaFree(0));
    CUDA_CHECK(cudaStreamCreate(&mState.stream));

    OPTIX_CHECK(optixInit());
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &context_log_cb;
    if (mEnableValidation)
    {
        options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
        options.logCallbackLevel = 4;
    }
    else
    {
        options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF;
        options.logCallbackLevel = 2; // error
    }
    CUcontext cu_ctx = 0; // zero means take the current context
    OPTIX_CHECK(optixDeviceContextCreate(cu_ctx, &options, &mState.context));

    mState.mParamsBuffer.reset(new OptixBuffer(sizeof(Params)));
}

bool OptiXRender::compactAccel(CUdeviceptr& buffer,
                               OptixTraversableHandle& handle,
                               CUdeviceptr result,
                               size_t outputSizeInBytes)
{
    // Get compacted size from device
    size_t compactedSize;
    CUDA_CHECK(cudaMemcpy(&compactedSize, (void*)result, sizeof(size_t), cudaMemcpyDeviceToHost));

    // Only compact if it saves space
    if (compactedSize >= outputSizeInBytes)
    {
        return false;
    }

    // Allocate compacted buffer
    CUdeviceptr compactedBuffer;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&compactedBuffer), compactedSize));

    // Compact acceleration structure into new buffer
    OPTIX_CHECK(optixAccelCompact(mState.context, 0, handle, compactedBuffer, compactedSize, &handle));

    // Free original buffer and update pointer
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(buffer)));
    buffer = compactedBuffer;

    return true;
}

OptiXRender::Curve* OptiXRender::createCurve(const oka::Curve& curve)
{
    Curve* rcurve = new Curve();
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS |
                               OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    const uint32_t pointsCount = mScene->getCurvesPoint().size(); // total points count in points buffer
    const int degree = 3;

    // each oka::Curves could contains many curves
    const uint32_t numCurves = curve.mVertexCountsCount;

    std::vector<int> segmentIndices;
    uint32_t offsetInsideCurveArray = 0;
    for (int curveIndex = 0; curveIndex < numCurves; ++curveIndex)
    {
        const std::vector<uint32_t>& vertexCounts = mScene->getCurvesVertexCounts();
        const uint32_t numControlPoints = vertexCounts[curve.mVertexCountsStart + curveIndex];
        const int segmentsCount = numControlPoints - degree;
        for (int i = 0; i < segmentsCount; ++i)
        {
            int index = curve.mPointsStart + offsetInsideCurveArray + i;
            segmentIndices.push_back(index);
        }
        offsetInsideCurveArray += numControlPoints;
    }

    const size_t segmentIndicesSize = sizeof(int) * segmentIndices.size();
    CUdeviceptr d_segmentIndices = 0;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_segmentIndices), segmentIndicesSize));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_segmentIndices), segmentIndices.data(), segmentIndicesSize, cudaMemcpyHostToDevice));
    // Curve build input.
    OptixBuildInput curve_input = {};

    curve_input.type = OPTIX_BUILD_INPUT_TYPE_CURVES;
    switch (degree)
    {
    case 1:
        curve_input.curveArray.curveType = OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR;
        break;
    case 2:
        curve_input.curveArray.curveType = OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE;
        break;
    case 3:
        curve_input.curveArray.curveType = OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE;
        break;
    }

    curve_input.curveArray.numPrimitives = segmentIndices.size();
    curve_input.curveArray.vertexBuffers = &d_points;
    curve_input.curveArray.numVertices = pointsCount;
    curve_input.curveArray.vertexStrideInBytes = sizeof(glm::float3);
    curve_input.curveArray.widthBuffers = &d_widths;
    curve_input.curveArray.widthStrideInBytes = sizeof(float);
    curve_input.curveArray.normalBuffers = 0;
    curve_input.curveArray.normalStrideInBytes = 0;
    curve_input.curveArray.indexBuffer = d_segmentIndices;
    curve_input.curveArray.indexStrideInBytes = sizeof(int);
    curve_input.curveArray.flag = OPTIX_GEOMETRY_FLAG_NONE;
    curve_input.curveArray.primitiveIndexOffset = 0;

    // curve_input.curveArray.endcapFlags = OPTIX_CURVE_ENDCAP_ON;

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(mState.context, &accel_options, &curve_input,
                                             1, // Number of build inputs
                                             &gas_buffer_sizes));

    CUdeviceptr d_temp_buffer_gas;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer_gas), gas_buffer_sizes.tempSizeInBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&rcurve->d_gas_output_buffer), gas_buffer_sizes.outputSizeInBytes));

    CUdeviceptr compactedSizeBuffer;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>((&compactedSizeBuffer)), sizeof(uint64_t)));

    OptixAccelEmitDesc property = {};
    property.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    property.result = compactedSizeBuffer;

    OPTIX_CHECK(optixAccelBuild(mState.context, mState.stream,
                                &accel_options, &curve_input,
                                1, // num build inputs
                                d_temp_buffer_gas, gas_buffer_sizes.tempSizeInBytes, rcurve->d_gas_output_buffer,
                                gas_buffer_sizes.outputSizeInBytes, &rcurve->gas_handle,
                                &property, // emitted property list
                                1)); // num emitted properties

    compactAccel(rcurve->d_gas_output_buffer, rcurve->gas_handle, property.result, gas_buffer_sizes.outputSizeInBytes);

    // We can now free the scratch space buffer used during build and the vertex
    // inputs, since they are not needed by our trivial shading method
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer_gas)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_segmentIndices)));

    return rcurve;
}

OptiXRender::Mesh* OptiXRender::createMesh(const oka::Mesh& mesh)
{
    OptixTraversableHandle gas_handle;
    CUdeviceptr d_gas_output_buffer;

    // Configure acceleration structure build options
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    // Set up triangle input data
    const CUdeviceptr vertexBuffer = mVertexBuffer->getPtr() + mesh.mVbOffset * sizeof(oka::Scene::Vertex);
    const CUdeviceptr indexBuffer = mIndexBuffer->getPtr() + mesh.mIndex * sizeof(uint32_t);

    // Our build input is a simple list of non-indexed triangle vertices
    const uint32_t triangle_input_flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
    OptixBuildInput triangle_input = {};
    triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangle_input.triangleArray.numVertices = mesh.mVertexCount;
    triangle_input.triangleArray.vertexBuffers = &vertexBuffer;
    triangle_input.triangleArray.vertexStrideInBytes = sizeof(oka::Scene::Vertex);
    triangle_input.triangleArray.indexBuffer = indexBuffer;
    triangle_input.triangleArray.indexFormat = OptixIndicesFormat::OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangle_input.triangleArray.indexStrideInBytes = sizeof(uint32_t) * 3;
    triangle_input.triangleArray.numIndexTriplets = mesh.mCount / 3;
    triangle_input.triangleArray.flags = triangle_input_flags;
    triangle_input.triangleArray.numSbtRecords = 1;

    // Calculate memory requirements
    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(mState.context, &accel_options, &triangle_input, 1, &gas_buffer_sizes));

    // Allocate required buffers
    CUdeviceptr d_temp_buffer_gas;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer_gas), gas_buffer_sizes.tempSizeInBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gas_output_buffer), gas_buffer_sizes.outputSizeInBytes));

    // Set up compaction
    CUdeviceptr compactedSizeBuffer;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&compactedSizeBuffer), sizeof(uint64_t)));

    OptixAccelEmitDesc property = {};
    property.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    property.result = compactedSizeBuffer;

    // Build acceleration structure
    OPTIX_CHECK(optixAccelBuild(mState.context,
                                mState.stream,
                                &accel_options, &triangle_input,
                                1, // num build inputs
                                d_temp_buffer_gas, gas_buffer_sizes.tempSizeInBytes, d_gas_output_buffer,
                                gas_buffer_sizes.outputSizeInBytes, &gas_handle,
                                &property, // emitted property list
                                1)); // num emitted properties

    // Compact the acceleration structure
    compactAccel(d_gas_output_buffer, gas_handle, property.result, gas_buffer_sizes.outputSizeInBytes);

    // Free temporary buffers
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer_gas)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(compactedSizeBuffer)));

    // Create and return mesh object
    Mesh* rmesh = new Mesh();
    rmesh->d_gas_output_buffer = d_gas_output_buffer;
    rmesh->gas_handle = gas_handle;
    return rmesh;
}

void OptiXRender::createBottomLevelAccelerationStructures()
{
    const std::vector<oka::Mesh>& meshes = mScene->getMeshes();
    const std::vector<oka::Curve>& curves = mScene->getCurves();
    // TODO: add proper clear and free resources
    mOptixMeshes.clear();
    for (int i = 0; i < meshes.size(); ++i)
    {
        Mesh* m = createMesh(meshes[i]);
        mOptixMeshes.emplace_back(m);
    }
    mOptixCurves.clear();
    for (int i = 0; i < curves.size(); ++i)
    {
        Curve* c = createCurve(curves[i]);
        mOptixCurves.emplace_back(c);
    }
}

void OptiXRender::createTopLevelAccelerationStructure()
{
    const std::vector<oka::Instance>& instances = mScene->getInstances();

    // Create OptixInstances from scene instances
    std::vector<OptixInstance> optixInstances;
    optixInstances.reserve(instances.size());

    for (const auto& instance : instances)
    {
        OptixInstance oi = {};
        
        // Set traversable handle and visibility mask based on instance type
        switch (instance.type)
        {
        case oka::Instance::Type::eMesh:
            oi.traversableHandle = mOptixMeshes[instance.mMeshId]->gas_handle;
            oi.visibilityMask = GEOMETRY_MASK_TRIANGLE;
            break;
        case oka::Instance::Type::eCurve:
            oi.traversableHandle = mOptixCurves[instance.mCurveId]->gas_handle;
            oi.visibilityMask = GEOMETRY_MASK_CURVE;
            break;
        case oka::Instance::Type::eLight:
            oi.traversableHandle = mOptixMeshes[instance.mMeshId]->gas_handle;
            oi.visibilityMask = GEOMETRY_MASK_LIGHT;
            break;
        default:
            STRELKA_ERROR("Unknown instance type");
            assert(0);
            break;
        }

        // Set transform and SBT offset
        memcpy(oi.transform, glm::value_ptr(glm::float3x4(glm::rowMajor4(instance.transform))), sizeof(float) * 12);
        oi.sbtOffset = static_cast<unsigned int>(optixInstances.size() * RAY_TYPE_COUNT);
        
        optixInstances.push_back(oi);
    }

    // Allocate/reallocate device memory for instances if needed
    const size_t instancesSize = sizeof(OptixInstance) * optixInstances.size();
    if (instancesSize != mState.d_instances_size)
    {
        if (mState.d_instances)
        {
            CUDA_CHECK(cudaFree(reinterpret_cast<void*>(mState.d_instances)));
        }
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&mState.d_instances), instancesSize));
        mState.d_instances_size = instancesSize;
    }

    // Copy instances to device
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(mState.d_instances), optixInstances.data(), instancesSize, cudaMemcpyHostToDevice));

    // Setup IAS build input
    OptixBuildInput iasInput = {};
    iasInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    iasInput.instanceArray.instances = mState.d_instances;
    iasInput.instanceArray.numInstances = static_cast<int>(optixInstances.size());

    // Setup IAS build options
    OptixAccelBuildOptions iasOptions = {};
    iasOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    iasOptions.motionOptions.numKeys = 1;
    iasOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    // Compute memory requirements
    OptixAccelBufferSizes iasBufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(mState.context, &iasOptions, &iasInput, 1, &iasBufferSizes));

    // Allocate buffers
    CUdeviceptr outputBuffer, tempBuffer, compactedSizeBuffer;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&outputBuffer), iasBufferSizes.outputSizeInBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&tempBuffer), iasBufferSizes.tempSizeInBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&compactedSizeBuffer), sizeof(uint64_t)));

    // Setup compaction property
    OptixAccelEmitDesc property = {};
    property.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    property.result = compactedSizeBuffer;

    // Build IAS
    OPTIX_CHECK(optixAccelBuild(
        mState.context, 
        mState.stream,
        &iasOptions,
        &iasInput,
        1, // num build inputs
        tempBuffer,
        iasBufferSizes.tempSizeInBytes,
        outputBuffer,
        iasBufferSizes.outputSizeInBytes,
        &mState.ias_handle,
        &property,
        1 // num emitted properties
    ));

    // Compact acceleration structure
    compactAccel(outputBuffer, mState.ias_handle, property.result, iasBufferSizes.outputSizeInBytes);

    // Cleanup temporary buffers
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(compactedSizeBuffer)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(tempBuffer)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(outputBuffer)));
}

void OptiXRender::createModule()
{
    OptixModule module = nullptr;
    OptixPipelineCompileOptions pipeline_compile_options = {};
    OptixModuleCompileOptions module_compile_options = {};

    {
        if (mEnableValidation)
        {
            module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
            module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
        }
        else
        {
            module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
            module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
        }

        pipeline_compile_options.usesMotionBlur = false;
        pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
        pipeline_compile_options.numPayloadValues = 2;
        pipeline_compile_options.numAttributeValues = 2;

        if (mEnableValidation) // Enables debug exceptions during optix launches. This may incur
            // significant performance cost and should only be done during
            // development.
            pipeline_compile_options.exceptionFlags =
                OPTIX_EXCEPTION_FLAG_USER | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
        else
        {
            pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
        }

        pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
        pipeline_compile_options.usesPrimitiveTypeFlags =
            OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE | OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_CUBIC_BSPLINE;

        size_t inputSize = 0;
        std::string optixSource;
        const fs::path cwdPath = fs::current_path();
        const fs::path precompiledOptixPath = cwdPath / "optix/render_generated_OptixRender.cu.optixir";

        readSourceFile(optixSource, precompiledOptixPath);
        const char* input = optixSource.c_str();
        inputSize = optixSource.size();
        char log[2048]; // For error reporting from OptiX creation functions
        size_t sizeof_log = sizeof(log);

        OPTIX_CHECK_LOG(optixModuleCreate(mState.context, &module_compile_options, &pipeline_compile_options, input,
                                          inputSize, log, &sizeof_log, &module));
    }
    mState.ptx_module = module;
    mState.pipeline_compile_options = pipeline_compile_options;
    mState.module_compile_options = module_compile_options;

    // hair modules
    OptixBuiltinISOptions builtinISOptions = {};
    builtinISOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
    // builtinISOptions.curveEndcapFlags = OPTIX_CURVE_ENDCAP_ON;

    builtinISOptions.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE;
    OPTIX_CHECK(optixBuiltinISModuleGet(mState.context, &mState.module_compile_options, &mState.pipeline_compile_options,
                                        &builtinISOptions, &mState.m_catromCurveModule));
}

OptixProgramGroup OptiXRender::createRadianceClosestHitProgramGroup(PathTracerState& state,
                                                                    char const* module_code,
                                                                    size_t module_size)
{
    char log[2048];
    size_t sizeof_log = sizeof(log);

    OptixModule mat_module = nullptr;
    OPTIX_CHECK_LOG(optixModuleCreate(state.context, &state.module_compile_options, &state.pipeline_compile_options,
                                      module_code, module_size, log, &sizeof_log, &mat_module));

    OptixProgramGroupOptions program_group_options = {};

    OptixProgramGroupDesc hit_prog_group_desc = {};
    hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hit_prog_group_desc.hitgroup.moduleCH = mat_module;
    hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";

    hit_prog_group_desc.hitgroup.moduleIS = mState.m_catromCurveModule;
    hit_prog_group_desc.hitgroup.entryFunctionNameIS = 0; // automatically supplied for built-in module

    sizeof_log = sizeof(log);
    OptixProgramGroup ch_hit_group = nullptr;
    OPTIX_CHECK_LOG(optixProgramGroupCreate(state.context, &hit_prog_group_desc,
                                            /*numProgramGroups=*/1, &program_group_options, log, &sizeof_log,
                                            &ch_hit_group));

    return ch_hit_group;
}

void OptiXRender::createProgramGroups()
{
    OptixProgramGroupOptions program_group_options = {}; // Initialize to zeros

    OptixProgramGroupDesc raygen_prog_group_desc = {}; //
    raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_prog_group_desc.raygen.module = mState.ptx_module;
    raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
    char log[2048]; // For error reporting from OptiX creation functions
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(mState.context, &raygen_prog_group_desc,
                                            1, // num program groups
                                            &program_group_options, log, &sizeof_log, &mState.raygen_prog_group));

    OptixProgramGroupDesc miss_prog_group_desc = {};
    miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module = mState.ptx_module;
    miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
    sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(mState.context, &miss_prog_group_desc,
                                            1, // num program groups
                                            &program_group_options, log, &sizeof_log, &mState.radiance_miss_group));

    memset(&miss_prog_group_desc, 0, sizeof(OptixProgramGroupDesc));
    miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module = nullptr; // NULL miss program for occlusion rays
    miss_prog_group_desc.miss.entryFunctionName = nullptr;
    sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(mState.context, &miss_prog_group_desc,
                                            1, // num program groups
                                            &program_group_options, log, &sizeof_log, &mState.occlusion_miss_group));

    OptixProgramGroupDesc hit_prog_group_desc = {};
    hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hit_prog_group_desc.hitgroup.moduleCH = mState.ptx_module;
    hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
    sizeof_log = sizeof(log);
    OptixProgramGroup radiance_hit_group;
    OPTIX_CHECK_LOG(optixProgramGroupCreate(mState.context, &hit_prog_group_desc,
                                            1, // num program groups
                                            &program_group_options, log, &sizeof_log, &radiance_hit_group));
    mState.radiance_default_hit_group = radiance_hit_group;

    OptixProgramGroupDesc light_hit_prog_group_desc = {};
    light_hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    light_hit_prog_group_desc.hitgroup.moduleCH = mState.ptx_module;
    light_hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__light";
    sizeof_log = sizeof(log);
    OptixProgramGroup light_hit_group;
    OPTIX_CHECK_LOG(optixProgramGroupCreate(mState.context, &light_hit_prog_group_desc,
                                            1, // num program groups
                                            &program_group_options, log, &sizeof_log, &light_hit_group));
    mState.light_hit_group = light_hit_group;

    memset(&hit_prog_group_desc, 0, sizeof(OptixProgramGroupDesc));
    hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hit_prog_group_desc.hitgroup.moduleCH = mState.ptx_module;
    hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__occlusion";

    hit_prog_group_desc.hitgroup.moduleIS = mState.m_catromCurveModule;
    hit_prog_group_desc.hitgroup.entryFunctionNameIS = 0; // automatically supplied for built-in module

    sizeof_log = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(mState.context, &hit_prog_group_desc,
                                        1, // num program groups
                                        &program_group_options, log, &sizeof_log, &mState.occlusion_hit_group));
}

void OptiXRender::createPipeline()
{
    OptixPipeline pipeline = nullptr;
    const uint32_t max_trace_depth = 2;
    std::vector<OptixProgramGroup> program_groups = {};

    program_groups.push_back(mState.raygen_prog_group);
    program_groups.push_back(mState.radiance_miss_group);
    program_groups.push_back(mState.radiance_default_hit_group);
    program_groups.push_back(mState.occlusion_miss_group);
    program_groups.push_back(mState.occlusion_hit_group);
    program_groups.push_back(mState.light_hit_group);

    for (auto& m : mMaterials)
    {
        program_groups.push_back(m.programGroup);
    }

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = max_trace_depth;

    char log[2048]; // For error reporting from OptiX creation functions
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixPipelineCreate(mState.context, &mState.pipeline_compile_options, &pipeline_link_options,
                                        program_groups.data(), program_groups.size(), log, &sizeof_log, &pipeline));

    OptixStackSizes stack_sizes = {};
    for (auto& prog_group : program_groups)
    {
        OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes, pipeline));
    }

    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes, max_trace_depth,
                                           0, // maxCCDepth
                                           0, // maxDCDepth
                                           &direct_callable_stack_size_from_traversal,
                                           &direct_callable_stack_size_from_state, &continuation_stack_size));
    OPTIX_CHECK(optixPipelineSetStackSize(pipeline, direct_callable_stack_size_from_traversal,
                                          direct_callable_stack_size_from_state, continuation_stack_size,
                                          2 // maxTraversableDepth
                                          ));
    mState.pipeline = pipeline;
}

void OptiXRender::createSbt()
{
    // TODO: add clear sbt if needed
    OptixShaderBindingTable sbt = {};
    CUdeviceptr raygen_record;
    const size_t raygen_record_size = sizeof(RayGenSbtRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&raygen_record), raygen_record_size));
    RayGenSbtRecord rg_sbt;
    OPTIX_CHECK(optixSbtRecordPackHeader(mState.raygen_prog_group, &rg_sbt));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(raygen_record), &rg_sbt, raygen_record_size, cudaMemcpyHostToDevice));

    CUdeviceptr miss_record;
    const uint32_t miss_record_count = RAY_TYPE_COUNT;
    size_t miss_record_size = sizeof(MissSbtRecord) * miss_record_count;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&miss_record), miss_record_size));
    std::vector<MissSbtRecord> missGroupDataCpu(miss_record_count);

    MissSbtRecord& ms_sbt = missGroupDataCpu[RAY_TYPE_RADIANCE];
    ms_sbt.data.bg_color = { 0.0f, 0.0f, 0.0f };
    OPTIX_CHECK(optixSbtRecordPackHeader(mState.radiance_miss_group, &ms_sbt));

    MissSbtRecord& ms_sbt_occlusion = missGroupDataCpu[RAY_TYPE_OCCLUSION];
    ms_sbt_occlusion.data = { 0.0f, 0.0f, 0.0f };
    OPTIX_CHECK(optixSbtRecordPackHeader(mState.occlusion_miss_group, &ms_sbt_occlusion));

    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(miss_record), missGroupDataCpu.data(), miss_record_size, cudaMemcpyHostToDevice));

    uint32_t hitGroupRecordCount = RAY_TYPE_COUNT;
    CUdeviceptr hitgroup_record = 0;
    size_t hitgroup_record_size = hitGroupRecordCount * sizeof(HitGroupSbtRecord);
    const std::vector<oka::Instance>& instances = mScene->getInstances();
    std::vector<HitGroupSbtRecord> hitGroupDataCpu(hitGroupRecordCount); // default sbt record
    if (!instances.empty())
    {
        const std::vector<oka::Mesh>& meshes = mScene->getMeshes();
        hitGroupRecordCount = instances.size() * RAY_TYPE_COUNT;
        hitgroup_record_size = sizeof(HitGroupSbtRecord) * hitGroupRecordCount;
        hitGroupDataCpu.resize(hitGroupRecordCount);
        for (int i = 0; i < instances.size(); ++i)
        {
            const oka::Instance& instance = instances[i];
            int hg_index = i * RAY_TYPE_COUNT + RAY_TYPE_RADIANCE;
            HitGroupSbtRecord& hg_sbt = hitGroupDataCpu[hg_index];
            // replace unknown material on default
            const int materialIdx = instance.mMaterialId == -1 ? 0 : instance.mMaterialId;
            const Material& mat = getMaterial(materialIdx);
            if (instance.type == oka::Instance::Type::eLight)
            {
                hg_sbt.data.lightId = instance.mLightId;
                const OptixProgramGroup& lightPg = mState.light_hit_group;
                OPTIX_CHECK(optixSbtRecordPackHeader(lightPg, &hg_sbt));
            }
            else
            {
                const OptixProgramGroup& hitMaterial = mat.programGroup;
                OPTIX_CHECK(optixSbtRecordPackHeader(hitMaterial, &hg_sbt));
            }
            // write all needed data for instances
            hg_sbt.data.argData = mat.d_argData;
            hg_sbt.data.roData = mat.d_roData;
            hg_sbt.data.resHandler = mat.d_textureHandler;
            if (instance.type == oka::Instance::Type::eMesh)
            {
                const oka::Mesh& mesh = meshes[instance.mMeshId];
                hg_sbt.data.indexCount = mesh.mCount;
                hg_sbt.data.indexOffset = mesh.mIndex;
                hg_sbt.data.vertexOffset = mesh.mVbOffset;
                hg_sbt.data.lightId = -1;
            }

            memcpy(hg_sbt.data.object_to_world, glm::value_ptr(glm::float4x4(glm::rowMajor4(instance.transform))),
                   sizeof(float4) * 4);
            glm::mat4 world_to_object = glm::inverse(instance.transform);
            memcpy(hg_sbt.data.world_to_object, glm::value_ptr(glm::float4x4(glm::rowMajor4(world_to_object))),
                   sizeof(float4) * 4);

            // write data for visibility ray
            hg_index = i * RAY_TYPE_COUNT + RAY_TYPE_OCCLUSION;
            HitGroupSbtRecord& hg_sbt_occlusion = hitGroupDataCpu[hg_index];
            OPTIX_CHECK(optixSbtRecordPackHeader(mState.occlusion_hit_group, &hg_sbt_occlusion));
        }
    }
    else
    {
        // stub record
        HitGroupSbtRecord& hg_sbt = hitGroupDataCpu[RAY_TYPE_RADIANCE];
        OPTIX_CHECK(optixSbtRecordPackHeader(mState.radiance_default_hit_group, &hg_sbt));
        HitGroupSbtRecord& hg_sbt_occlusion = hitGroupDataCpu[RAY_TYPE_OCCLUSION];
        OPTIX_CHECK(optixSbtRecordPackHeader(mState.occlusion_hit_group, &hg_sbt_occlusion));
    }
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&hitgroup_record), hitgroup_record_size));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(hitgroup_record), hitGroupDataCpu.data(), hitgroup_record_size,
                          cudaMemcpyHostToDevice));

    sbt.raygenRecord = raygen_record;
    sbt.missRecordBase = miss_record;
    sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
    sbt.missRecordCount = RAY_TYPE_COUNT;
    sbt.hitgroupRecordBase = hitgroup_record;
    sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
    sbt.hitgroupRecordCount = hitGroupRecordCount;
    mState.sbt = sbt;
}

void OptiXRender::updatePathtracerParams(const uint32_t width, const uint32_t height)
{
    bool needRealloc = false;
    if (mState.params.image_width != width || mState.params.image_height != height)
    {
        // new dimensions!
        needRealloc = true;
        // reset rendering
        getSharedContext().mSubframeIndex = 0;
    }
    mState.params.image_width = width;
    mState.params.image_height = height;
    if (needRealloc)
    {
        getSettings()->setAs<bool>("render/pt/isResized", true);
        if (mState.params.accum)
        {
            CUDA_CHECK(cudaFree((void*)mState.params.accum));
        }
        if (mState.params.diffuse)
        {
            CUDA_CHECK(cudaFree((void*)mState.params.diffuse));
        }
        if (mState.params.specular)
        {
            CUDA_CHECK(cudaFree((void*)mState.params.specular));
        }
        const size_t frameSize = mState.params.image_width * mState.params.image_height;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&mState.params.accum), frameSize * sizeof(float4)));

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&mState.params.diffuse), frameSize * sizeof(float4)));
        CUDA_CHECK(cudaMemset(mState.params.diffuse, 0, frameSize * sizeof(float4)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&mState.params.diffuseCounter), frameSize * sizeof(uint16_t)));
        CUDA_CHECK(cudaMemset(mState.params.diffuseCounter, 0, frameSize * sizeof(uint16_t)));

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&mState.params.specular), frameSize * sizeof(float4)));
        CUDA_CHECK(cudaMemset(mState.params.specular, 0, frameSize * sizeof(float4)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&mState.params.specularCounter), frameSize * sizeof(uint16_t)));
        CUDA_CHECK(cudaMemset(mState.params.specularCounter, 0, frameSize * sizeof(uint16_t)));
    }
}

void OptiXRender::render(Buffer* output)
{
    if (getSharedContext().mFrameNumber == 0)
    {
        createOptixMaterials();
        createPipeline();
        createVertexBuffer();
        createIndexBuffer();
        // upload all curve data
        createPointsBuffer();
        createWidthsBuffer();
        createBottomLevelAccelerationStructures();
        createTopLevelAccelerationStructure();
        createSbt();
        createLightBuffer();
    }

    if (mScene->getDirtyState() == DirtyFlag::eLights)
    {
        createLightBuffer();
        createTopLevelAccelerationStructure();
        // createSbt();

        mScene->clearDirtyState();
    }

    const uint32_t width = output->width();
    const uint32_t height = output->height();

    updatePathtracerParams(width, height);

    oka::Camera& camera = mScene->getCamera(0);
    camera.updateAspectRatio(width / (float)height);
    camera.updateViewMatrix();

    View currView = {};

    currView.mCamMatrices = camera.matrices;

    if (glm::any(glm::notEqual(currView.mCamMatrices.perspective, mPrevView.mCamMatrices.perspective)) ||
        glm::any(glm::notEqual(currView.mCamMatrices.view, mPrevView.mCamMatrices.view)))
    {
        // need reset
        getSharedContext().mSubframeIndex = 0;
    }

    SettingsManager& settings = *getSettings();
    bool settingsChanged = false;

    static uint32_t rectLightSamplingMethodPrev = 0;
    const uint32_t rectLightSamplingMethod = settings.getAs<uint32_t>("render/pt/rectLightSamplingMethod");
    settingsChanged = (rectLightSamplingMethodPrev != rectLightSamplingMethod);
    rectLightSamplingMethodPrev = rectLightSamplingMethod;

    static bool enableAccumulationPrev = 0;
    bool enableAccumulation = settings.getAs<bool>("render/pt/enableAcc");
    settingsChanged |= (enableAccumulationPrev != enableAccumulation);
    enableAccumulationPrev = enableAccumulation;

    static uint32_t sspTotalPrev = 0;
    const uint32_t sspTotal = settings.getAs<uint32_t>("render/pt/sppTotal");
    settingsChanged |= (sspTotalPrev > sspTotal); // reset only if new spp less than already accumulated
    sspTotalPrev = sspTotal;

    const float gamma = settings.getAs<float>("render/post/gamma");
    const ToneMapperType tonemapperType = (ToneMapperType)settings.getAs<uint32_t>("render/pt/tonemapperType");

    if (settingsChanged)
    {
        getSharedContext().mSubframeIndex = 0;
    }

    Params& params = mState.params;
    params.scene.vb = (Vertex*)mVertexBuffer->getPtr();
    params.scene.ib = (uint32_t*)mIndexBuffer->getPtr();
    params.scene.lights = (UniformLight*)mLightBuffer->getPtr();
    params.scene.numLights = mScene->getLights().size();

    params.image = (float4*)((OptixBuffer*)output)->getNativePtr();
    params.samples_per_launch = settings.getAs<uint32_t>("render/pt/spp");
    params.handle = mState.ias_handle;
    params.max_depth = settings.getAs<uint32_t>("render/pt/depth");

    params.rectLightSamplingMethod = settings.getAs<uint32_t>("render/pt/rectLightSamplingMethod");
    params.enableAccumulation = settings.getAs<bool>("render/pt/enableAcc");
    params.debug = settings.getAs<uint32_t>("render/pt/debug");
    params.shadowRayTmin = settings.getAs<float>("render/pt/dev/shadowRayTmin");
    params.materialRayTmin = settings.getAs<float>("render/pt/dev/materialRayTmin");

    memcpy(params.viewToWorld, glm::value_ptr(glm::transpose(glm::inverse(camera.matrices.view))),
           sizeof(params.viewToWorld));
    memcpy(params.clipToView, glm::value_ptr(glm::transpose(camera.matrices.invPerspective)), sizeof(params.clipToView));
    params.subframe_index = getSharedContext().mSubframeIndex;
    // Photometric Units from iray documentation
    // Controls the sensitivity of the "camera film" and is expressed as an index; the ISO number of the film, also
    // known as "film speed." The higher this value, the greater the exposure. If this is set to a non-zero value,
    // "Photographic" mode is enabled. If this is set to 0, "Arbitrary" mode is enabled, and all color scaling is then
    // strictly defined by the value of cm^2 Factor.
    float filmIso = settings.getAs<float>("render/post/tonemapper/filmIso");
    // The candela per meter square factor
    float cm2_factor = settings.getAs<float>("render/post/tonemapper/cm2_factor");
    // The fractional aperture number; e.g., 11 means aperture "f/11." It adjusts the size of the opening of the "camera
    // iris" and is expressed as a ratio. The higher this value, the lower the exposure.
    float fStop = settings.getAs<float>("render/post/tonemapper/fStop");
    // Controls the duration, in fractions of a second, that the "shutter" is open; e.g., the value 100 means that the
    // "shutter" is open for 1/100th of a second. The higher this value, the greater the exposure
    float shutterSpeed = settings.getAs<float>("render/post/tonemapper/shutterSpeed");
    // Specifies the main color temperature of the light sources; the color that will be mapped to "white" on output,
    // e.g., an incoming color of this hue/saturation will be mapped to grayscale, but its intensity will remain
    // unchanged. This is similar to white balance controls on digital cameras.
    float3 whitePoint{ 1.0f, 1.0f, 1.0f };
    float3 exposureValue = all(whitePoint) ? 1.0f / whitePoint : make_float3(1.0f);
    const float lum = dot(exposureValue, make_float3(0.299f, 0.587f, 0.114f));
    if (filmIso > 0.0f)
    {
        // See https://www.nayuki.io/page/the-photographic-exposure-equation
        exposureValue *= cm2_factor * filmIso / (shutterSpeed * fStop * fStop) / 100.0f;
    }
    else
    {
        exposureValue *= cm2_factor;
    }
    exposureValue /= lum;

    params.exposure = exposureValue;

    const uint32_t totalSpp = settings.getAs<uint32_t>("render/pt/sppTotal");
    const uint32_t samplesPerLaunch = settings.getAs<uint32_t>("render/pt/spp");
    const int32_t leftSpp = totalSpp - getSharedContext().mSubframeIndex;
    // if accumulation is off then launch selected samples per pixel
    uint32_t samplesThisLaunch = enableAccumulation ? std::min((int32_t)samplesPerLaunch, leftSpp) : samplesPerLaunch;
    // not to trace rays if there is no geometry
    if (mScene->getIndices().empty())
    {
        samplesThisLaunch = 0;
    }

    if (params.debug == 1)
    {
        samplesThisLaunch = 1;
        enableAccumulation = false;
    }

    params.samples_per_launch = samplesThisLaunch;
    params.enableAccumulation = enableAccumulation;
    params.maxSampleCount = totalSpp;

    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(mState.mParamsBuffer->getPtr()), &params, sizeof(params), cudaMemcpyHostToDevice));

    if (samplesThisLaunch != 0)
    {
        // Launch OptiX path tracer
        OPTIX_CHECK(optixLaunch(
            mState.pipeline,
            mState.stream,
            mState.mParamsBuffer->getPtr(),
            sizeof(Params),
            &mState.sbt,
            width,
            height,
            /*depth=*/1));
        CUDA_SYNC_CHECK();

        // Update subframe index for accumulation
        getSharedContext().mSubframeIndex = enableAccumulation ? 
            getSharedContext().mSubframeIndex + samplesThisLaunch : 0;
    }
    else
    {
        // Copy accumulated buffer to output image
        const size_t imageSize = mState.params.image_width * mState.params.image_height * sizeof(float4);
        const void* srcBuffer = nullptr;

        // Select source buffer based on debug mode
        switch (params.debug) {
            case 0:  srcBuffer = params.accum;    break;
            case 2:  srcBuffer = params.diffuse;  break;
            case 3:  srcBuffer = params.specular; break;
        }

        if (srcBuffer) {
            CUDA_CHECK(cudaMemcpy(params.image, srcBuffer, imageSize, cudaMemcpyDeviceToDevice));
        }
    }

    // Apply tonemapping except for debug mode 1
    if (params.debug != 1)
    {
        float maxEDR = settings.getAs<float>("render/post/tonemapper/maxEDR");
        exposureValue *= maxEDR;
        tonemap(tonemapperType, exposureValue, gamma, params.image, width, height);
    }

    getSharedContext().mFrameNumber++;

    mPrevView = currView;
    mState.prevParams = mState.params;
}

void OptiXRender::init()
{
    // TODO: move USD_DIR to settings
    const char* envUSDPath = std::getenv("USD_DIR");
    mEnableValidation = getSettings()->getAs<bool>("render/enableValidation");

    fs::path usdMdlLibPath;
    if (envUSDPath)
    {
        usdMdlLibPath = (fs::path(envUSDPath) / fs::path("libraries/mdl/")).make_preferred();
    }
    const fs::path cwdPath = fs::current_path();
    STRELKA_DEBUG("cwdPath: {}", cwdPath.string().c_str());
    const fs::path mtlxPath = (cwdPath / fs::path("data/materials/mtlx")).make_preferred();
    STRELKA_DEBUG("mtlxPath: {}", mtlxPath.string().c_str());
    const fs::path mdlPath = (cwdPath / fs::path("data/materials/mdl")).make_preferred();

    const std::string usdMdlLibPathStr = usdMdlLibPath.string();
    const std::string mtlxPathStr = mtlxPath.string().c_str();
    const std::string mdlPathStr = mdlPath.string().c_str();

    const char* paths[] = { usdMdlLibPathStr.c_str(), mtlxPathStr.c_str(), mdlPathStr.c_str() };
    bool res = mMaterialManager.addMdlSearchPath(paths, sizeof(paths) / sizeof(char*));

    if (!res)
    {
        STRELKA_FATAL("Wrong mdl paths configuration!");
        assert(0);
        return;
    }

    // default material
    {
        oka::Scene::MaterialDescription defaultMaterial{};
        defaultMaterial.file = "default.mdl";
        defaultMaterial.name = "default_material";
        defaultMaterial.type = oka::Scene::MaterialDescription::Type::eMdl;
        mScene->addMaterial(defaultMaterial);
    }

    createContext();
    // createAccelerationStructure();
    createModule();
    createProgramGroups();
    createPipeline();
    // createSbt();
}

Buffer* OptiXRender::createBuffer(const BufferDesc& desc)
{
    const size_t size = desc.height * desc.width * Buffer::getElementSize(desc.format);
    assert(size != 0);
    void* devicePtr = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&devicePtr), size));
    OptixBuffer* res = new OptixBuffer(devicePtr, desc.format, desc.width, desc.height);
    return res;
}

void OptiXRender::createPointsBuffer()
{
    const std::vector<glm::float3>& points = mScene->getCurvesPoint();

    std::vector<float3> data(points.size());
    for (int i = 0; i < points.size(); ++i)
    {
        data[i] = make_float3(points[i].x, points[i].y, points[i].z);
    }
    const size_t size = data.size() * sizeof(float3);

    if (d_points)
    {
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_points)));
    }
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_points), size));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_points), data.data(), size, cudaMemcpyHostToDevice));
}

void OptiXRender::createWidthsBuffer()
{
    const std::vector<float>& data = mScene->getCurvesWidths();
    const size_t size = data.size() * sizeof(float);

    if (d_widths)
    {
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_widths)));
    }
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_widths), size));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_widths), data.data(), size, cudaMemcpyHostToDevice));
}

void OptiXRender::createVertexBuffer()
{
    const std::vector<oka::Scene::Vertex>& vertices = mScene->getVertices();
    const size_t vbsize = vertices.size() * sizeof(oka::Scene::Vertex);

    if (mVertexBuffer == nullptr)
    {
        mVertexBuffer.reset(new OptixBuffer(vbsize));
    }
    if (mVertexBuffer->size() != vbsize)
    {
        mVertexBuffer->realloc(vbsize);
    }
    CUDA_CHECK(
        cudaMemcpy(reinterpret_cast<void*>(mVertexBuffer->getPtr()), vertices.data(), vbsize, cudaMemcpyHostToDevice));
}

void OptiXRender::createIndexBuffer()
{
    const std::vector<uint32_t>& indices = mScene->getIndices();
    const size_t ibsize = indices.size() * sizeof(uint32_t);

    if (mIndexBuffer == nullptr)
    {
        mIndexBuffer.reset(new OptixBuffer(ibsize));
    }
    if (mIndexBuffer->size() != ibsize)
    {
        mIndexBuffer->realloc(ibsize);
    }
    CUDA_CHECK(
        cudaMemcpy(reinterpret_cast<void*>(mIndexBuffer->getPtr()), indices.data(), ibsize, cudaMemcpyHostToDevice));
}

void OptiXRender::createLightBuffer()
{
    const std::vector<Scene::Light>& lightDescs = mScene->getLights();
    const size_t lightBufferSize = lightDescs.size() * sizeof(Scene::Light);
    if (mLightBuffer == nullptr)
    {
        mLightBuffer.reset(new OptixBuffer(lightBufferSize));
    }
    if (mLightBuffer->size() != lightBufferSize)
    {
        mLightBuffer->realloc(lightBufferSize);
    }
    if (lightBufferSize)
    {
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(mLightBuffer->getPtr()), lightDescs.data(), lightBufferSize,
                              cudaMemcpyHostToDevice));
    }
}

Texture OptiXRender::loadTextureFromFile(const std::string& fileName)
{
    int texWidth, texHeight, texChannels;
    stbi_uc* data = stbi_load(fileName.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
    if (!data)
    {
        STRELKA_ERROR("Unable to load texture from file: {}", fileName.c_str());
        return Texture();
    }
    // TODO: add compression here to save gpu mem

    const void* dataPtr = data;

    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<uchar4>();
    cudaResourceDesc res_desc{};
    memset(&res_desc, 0, sizeof(res_desc));

    cudaArray_t device_tex_array;
    CUDA_CHECK(cudaMallocArray(&device_tex_array, &channel_desc, texWidth, texHeight));

    CUDA_CHECK(cudaMemcpy2DToArray(device_tex_array, 0, 0, dataPtr, texWidth * sizeof(char) * 4,
                                   texWidth * sizeof(char) * 4, texHeight, cudaMemcpyHostToDevice));

    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = device_tex_array;

    // Create filtered texture object
    cudaTextureDesc tex_desc;
    memset(&tex_desc, 0, sizeof(tex_desc));
    cudaTextureAddressMode addr_mode = cudaAddressModeWrap;
    tex_desc.addressMode[0] = addr_mode;
    tex_desc.addressMode[1] = addr_mode;
    tex_desc.addressMode[2] = addr_mode;
    tex_desc.filterMode = cudaFilterModeLinear;
    tex_desc.readMode = cudaReadModeNormalizedFloat;
    tex_desc.normalizedCoords = 1;
    if (res_desc.resType == cudaResourceTypeMipmappedArray)
    {
        tex_desc.mipmapFilterMode = cudaFilterModeLinear;
        tex_desc.maxAnisotropy = 16;
        tex_desc.minMipmapLevelClamp = 0.f;
        tex_desc.maxMipmapLevelClamp = 1000.f; // default value in OpenGL
    }
    cudaTextureObject_t tex_obj = 0;
    CUDA_CHECK(cudaCreateTextureObject(&tex_obj, &res_desc, &tex_desc, nullptr));
    // Create unfiltered texture object if necessary (cube textures have no texel functions)
    cudaTextureObject_t tex_obj_unfilt = 0;
    // if (texture_shape != mi::neuraylib::ITarget_code::Texture_shape_cube)
    {
        // Use a black border for access outside of the texture
        tex_desc.addressMode[0] = cudaAddressModeBorder;
        tex_desc.addressMode[1] = cudaAddressModeBorder;
        tex_desc.addressMode[2] = cudaAddressModeBorder;
        tex_desc.filterMode = cudaFilterModePoint;

        CUDA_CHECK(cudaCreateTextureObject(&tex_obj_unfilt, &res_desc, &tex_desc, nullptr));
    }
    stbi_image_free(data);
    return Texture(tex_obj, tex_obj_unfilt, make_uint3(texWidth, texHeight, 1));
}

bool OptiXRender::createOptixMaterials()
{
    std::unordered_map<std::string, MaterialManager::Module*> nameToModule;
    std::unordered_map<std::string, MaterialManager::MaterialInstance*> nameToInstance;
    std::unordered_map<std::string, MaterialManager::CompiledMaterial*> nameToCompiled;

    std::unordered_map<std::string, MaterialManager::TargetCode*> nameToTargetCode;

    std::vector<MaterialManager::CompiledMaterial*> compiledMaterials;
    MaterialManager::TargetCode* targetCode;

    std::vector<Scene::MaterialDescription>& matDescs = mScene->getMaterials();
    for (uint32_t i = 0; i < matDescs.size(); ++i)
    {
        oka::Scene::MaterialDescription& currMatDesc = matDescs[i];
        if (currMatDesc.type == oka::Scene::MaterialDescription::Type::eMdl)
        {
            if (auto it = nameToCompiled.find(currMatDesc.name); it != nameToCompiled.end())
            {
                compiledMaterials.emplace_back(it->second);
                continue;
            }

            MaterialManager::Module* mdlModule = nullptr;
            if (nameToModule.find(currMatDesc.file) != nameToModule.end())
            {
                mdlModule = nameToModule[currMatDesc.file];
            }
            else
            {
                mdlModule = mMaterialManager.createModule(currMatDesc.file.c_str());
                if (mdlModule == nullptr)
                {
                    STRELKA_ERROR("Unable to load MDL file: {}, Force replaced to default.mdl", currMatDesc.file.c_str());
                    mdlModule = nameToModule["default.mdl"];
                }
                nameToModule[currMatDesc.file] = mdlModule;
            }
            assert(mdlModule);
            MaterialManager::MaterialInstance* materialInst = nullptr;
            if (nameToInstance.find(currMatDesc.name) != nameToInstance.end())
            {
                materialInst = nameToInstance[currMatDesc.name];
            }
            else
            {
                materialInst = mMaterialManager.createMaterialInstance(mdlModule, currMatDesc.name.c_str());
                nameToInstance[currMatDesc.name] = materialInst;
            }
            assert(materialInst);
            MaterialManager::CompiledMaterial* materialComp = nullptr;
            {
                materialComp = mMaterialManager.compileMaterial(materialInst);
                nameToCompiled[currMatDesc.name] = materialComp;
            }
            assert(materialComp);
            compiledMaterials.push_back(materialComp);
        }
        else
        {
            MaterialManager::Module* mdlModule = mMaterialManager.createMtlxModule(currMatDesc.code.c_str());
            assert(mdlModule);
            MaterialManager::MaterialInstance* materialInst = mMaterialManager.createMaterialInstance(mdlModule, "");
            assert(materialInst);
            MaterialManager::CompiledMaterial* materialComp = mMaterialManager.compileMaterial(materialInst);
            assert(materialComp);
            compiledMaterials.push_back(materialComp);
            // MaterialManager::TargetCode* mdlTargetCode = mMaterialManager.generateTargetCode(&materialComp, 1);
            // assert(mdlTargetCode);
            // mNameToTargetCode[currMatDesc.name] = mdlTargetCode;
            // targetCodes.push_back(mdlTargetCode);
        }
    }

    targetCode = mMaterialManager.generateTargetCode(compiledMaterials.data(), compiledMaterials.size());

    std::vector<Texture> materialTextures;

    const auto searchPath = getSettings()->getAs<std::string>("resource/searchPath");
    fs::path resourcePath = fs::path(getSettings()->getAs<std::string>("resource/searchPath"));

    for (uint32_t i = 0; i < matDescs.size(); ++i)
    {
        for (const auto& param : matDescs[i].params)
        {
            bool res = false;
            if (param.type == MaterialManager::Param::Type::eTexture)
            {
                std::string texPath(param.value.begin(), param.value.end());
                // int texId = getTexManager()->loadTextureMdl(texPath);
                fs::path fullTextureFilePath = resourcePath / texPath;
                ::Texture tex = loadTextureFromFile(fullTextureFilePath.string());
                materialTextures.push_back(tex);
                int texId = 0;
                int resId = mMaterialManager.registerResource(targetCode, texId);
                assert(resId > 0);
                MaterialManager::Param newParam;
                newParam.name = param.name;
                newParam.type = MaterialManager::Param::Type::eInt;
                newParam.value.resize(sizeof(resId));
                memcpy(newParam.value.data(), &resId, sizeof(resId));
                res = mMaterialManager.setParam(targetCode, i, compiledMaterials[i], newParam);
            }
            else
            {
                res = mMaterialManager.setParam(targetCode, i, compiledMaterials[i], param);
            }
            if (!res)
            {
                STRELKA_ERROR(
                    "Unable to set parameter: {0} for material: {1}", param.name.c_str(), matDescs[i].name.c_str());
                // assert(0);
            }
        }
        mMaterialManager.dumpParams(targetCode, i, compiledMaterials[i]);
    }

    const uint8_t* argData = mMaterialManager.getArgBufferData(targetCode);
    const size_t argDataSize = mMaterialManager.getArgBufferSize(targetCode);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_materialArgData), argDataSize));
    CUDA_CHECK(cudaMemcpy((void*)d_materialArgData, argData, argDataSize, cudaMemcpyHostToDevice));

    const uint8_t* roData = mMaterialManager.getReadOnlyBlockData(targetCode);
    const size_t roDataSize = mMaterialManager.getReadOnlyBlockSize(targetCode);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_materialRoData), roDataSize));
    CUDA_CHECK(cudaMemcpy((void*)d_materialRoData, roData, roDataSize, cudaMemcpyHostToDevice));

    const size_t texturesBuffSize = sizeof(Texture) * materialTextures.size();
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_texturesData), texturesBuffSize));
    CUDA_CHECK(cudaMemcpy((void*)d_texturesData, materialTextures.data(), texturesBuffSize, cudaMemcpyHostToDevice));

    Texture_handler resourceHandler;
    resourceHandler.num_textures = materialTextures.size();
    resourceHandler.textures = (const Texture*)d_texturesData;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_texturesHandler), sizeof(Texture_handler)));
    CUDA_CHECK(cudaMemcpy((void*)d_texturesHandler, &resourceHandler, sizeof(Texture_handler), cudaMemcpyHostToDevice));

    std::unordered_map<MaterialManager::CompiledMaterial*, OptixProgramGroup> compiledToOptixPG;
    for (int i = 0; i < compiledMaterials.size(); ++i)
    {
        if (compiledToOptixPG.find(compiledMaterials[i]) == compiledToOptixPG.end())
        {
            const char* codeData = mMaterialManager.getShaderCode(targetCode, i);
            assert(codeData);
            const size_t codeSize = strlen(codeData);
            assert(codeSize);
            OptixProgramGroup pg = createRadianceClosestHitProgramGroup(mState, codeData, codeSize);
            compiledToOptixPG[compiledMaterials[i]] = pg;
        }

        Material optixMaterial;
        optixMaterial.programGroup = compiledToOptixPG[compiledMaterials[i]];
        optixMaterial.d_argData = d_materialArgData + mMaterialManager.getArgBlockOffset(targetCode, i);
        optixMaterial.d_argDataSize = argDataSize;
        optixMaterial.d_roData = d_materialRoData + mMaterialManager.getReadOnlyOffset(targetCode, i);
        optixMaterial.d_roSize = roDataSize;
        optixMaterial.d_textureHandler = d_texturesHandler;

        mMaterials.push_back(optixMaterial);
    }

    // Clean up material resources
    for (auto& [name, module] : nameToModule) {
        mMaterialManager.destroyModule(module);
    }
    for (auto& [name, instance] : nameToInstance) {
        mMaterialManager.destroyMaterialInstance(instance);
    }
    // TODO: ... cleanup other material resources

    return true;
}

OptiXRender::Material& OptiXRender::getMaterial(int id)
{
    assert(id < mMaterials.size());
    return mMaterials[id];
}
