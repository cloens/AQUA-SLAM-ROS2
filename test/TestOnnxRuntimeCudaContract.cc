#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <onnxruntime_session_options_config_keys.h>

namespace
{

std::vector<std::string> Names(Ort::Session& session, bool inputs)
{
    Ort::AllocatorWithDefaultOptions allocator;
    const std::size_t count = inputs ? session.GetInputCount() : session.GetOutputCount();
    std::vector<std::string> names;
    names.reserve(count);
    for (std::size_t index = 0; index < count; ++index)
    {
        auto name = inputs ? session.GetInputNameAllocated(index, allocator)
                           : session.GetOutputNameAllocated(index, allocator);
        names.emplace_back(name.get());
    }
    return names;
}

void ExpectNames(const std::vector<std::string>& actual,
                 const std::vector<std::string>& expected,
                 const std::string& model)
{
    if (actual != expected)
    {
        std::cerr << model << " names mismatch. actual:";
        for (const auto& name : actual)
            std::cerr << ' ' << name;
        std::cerr << std::endl;
        throw std::runtime_error(model + " tensor contract mismatch");
    }
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cerr << "usage: TestOnnxRuntimeCudaContract SUPERPOINT LIGHTGLUE" << std::endl;
        return 2;
    }

    try
    {
        const auto providers = Ort::GetAvailableProviders();
        if (std::find(providers.begin(), providers.end(), "CUDAExecutionProvider") == providers.end())
            throw std::runtime_error("CUDAExecutionProvider is unavailable");

        Ort::Env env(
#ifdef AQUA_ORT_DIAGNOSTIC_CPU_FALLBACK
            ORT_LOGGING_LEVEL_VERBOSE,
#else
            ORT_LOGGING_LEVEL_WARNING,
#endif
            "aqua-neural-contract");
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(1);
        options.SetInterOpNumThreads(1);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
#ifdef AQUA_ORT_STRICT_NO_CPU_NODES
        options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1");
#endif
        OrtCUDAProviderOptions cuda_options{};
        cuda_options.device_id = 0;
        options.AppendExecutionProvider_CUDA(cuda_options);

        Ort::Session superpoint(env, argv[1], options);
        ExpectNames(Names(superpoint, true), {"image"}, "SuperPoint inputs");
        ExpectNames(Names(superpoint, false),
                    {"keypoints", "scores", "descriptors"},
                    "SuperPoint outputs");

        Ort::Session lightglue(env, argv[2], options);
        ExpectNames(Names(lightglue, true),
                    {"kpts0", "kpts1", "desc0", "desc1"},
                    "LightGlue inputs");
        ExpectNames(Names(lightglue, false), {"matches0", "mscores0"},
                    "LightGlue outputs");
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << std::endl;
        return 1;
    }
    return 0;
}
