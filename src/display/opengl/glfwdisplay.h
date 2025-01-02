#pragma once

#include <glad/glad.h>

#include "Display.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

namespace oka
{

class GlfwDisplay : public Display
{
private:
    // OpenGL texture for CUDA rendering
    cudaGraphicsResource* cudaResource;
    GLuint m_render_tex = 0u;

public:
    GlfwDisplay();
    virtual ~GlfwDisplay();

    virtual void init(int width, int height, SettingsManager* settings) override;
    void destroy();

    void onBeginFrame();
    void onEndFrame();

    void drawFrame(ImageBuffer& result);
    void drawUI();

    void* getDisplayNativeTexure() override;
    float getMaxEDR() override;
};
} // namespace oka
