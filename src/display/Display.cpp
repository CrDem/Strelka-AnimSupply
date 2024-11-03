#include "Display.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"

using namespace oka;

void Display::framebufferResizeCallback(GLFWwindow* window, int width, int height)
{
    assert(window);
    if (width == 0 || height == 0)
    {
        return;
    }

    auto app = reinterpret_cast<Display*>(glfwGetWindowUserPointer(window));
    // app->framebufferResized = true;
    ResizeHandler* handler = app->getResizeHandler();
    if (handler)
    {
        handler->framebufferResize(width, height);
    }
}

void Display::keyCallback(GLFWwindow* window,
                          [[maybe_unused]] int key,
                          [[maybe_unused]] int scancode,
                          [[maybe_unused]] int action,
                          [[maybe_unused]] int mods)
{
    assert(window);
    ImGuiIO& io = ImGui::GetIO();
    bool handled = io.WantCaptureKeyboard;
    if (handled)
    {
        return;
    }
    auto app = reinterpret_cast<Display*>(glfwGetWindowUserPointer(window));
    InputHandler* handler = app->getInputHandler();
    assert(handler);
    handler->keyCallback(key, scancode, action, mods);
}

void Display::mouseButtonCallback(GLFWwindow* window,
                                  [[maybe_unused]] int button,
                                  [[maybe_unused]] int action,
                                  [[maybe_unused]] int mods)
{
    assert(window);
    ImGuiIO& io = ImGui::GetIO();
    bool handled = io.WantCaptureMouse;
    if (handled)
    {
        return;
    }
    auto app = reinterpret_cast<Display*>(glfwGetWindowUserPointer(window));
    InputHandler* handler = app->getInputHandler();
    if (handler)
    {
        handler->mouseButtonCallback(button, action, mods);
    }
}

void Display::handleMouseMoveCallback(GLFWwindow* window, [[maybe_unused]] double xpos, [[maybe_unused]] double ypos)
{
    assert(window);
    ImGuiIO& io = ImGui::GetIO();
    bool handled = io.WantCaptureMouse;
    if (handled)
    {
        return;
    }
    auto app = reinterpret_cast<Display*>(glfwGetWindowUserPointer(window));
    InputHandler* handler = app->getInputHandler();
    if (handler)
    {
        handler->handleMouseMoveCallback(xpos, ypos);
    }
}

void Display::scrollCallback(GLFWwindow* window, [[maybe_unused]] double xoffset, [[maybe_unused]] double yoffset)
{
    assert(window);
}

void Display::drawUI()
{
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGuiIO& io = ImGui::GetIO();

    const char* debugItems[] = { "None", "Normals", "Diffuse AOV", "Specular AOV" };
    static int currentDebugItemId = 0;

    ImGui::Begin("Menu:"); // begin window

    if (ImGui::BeginCombo("Debug view", debugItems[currentDebugItemId]))
    {
        for (int n = 0; n < IM_ARRAYSIZE(debugItems); n++)
        {
            bool is_selected = (currentDebugItemId == n);
            if (ImGui::Selectable(debugItems[n], is_selected))
            {
                currentDebugItemId = n;
            }
            if (is_selected)
            {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }
    mSettings->setAs<uint32_t>("render/pt/debug", currentDebugItemId);

    if (ImGui::TreeNode("Path Tracer"))
    {
        const char* rectlightSamplingMethodItems[] = { "Uniform", "Advanced" };
        static int currentRectlightSamplingMethodItemId = 0;
        if (ImGui::BeginCombo("Rect Light Sampling", rectlightSamplingMethodItems[currentRectlightSamplingMethodItemId]))
        {
            for (int n = 0; n < IM_ARRAYSIZE(rectlightSamplingMethodItems); n++)
            {
                bool is_selected = (currentRectlightSamplingMethodItemId == n);
                if (ImGui::Selectable(rectlightSamplingMethodItems[n], is_selected))
                {
                    currentRectlightSamplingMethodItemId = n;
                }
                if (is_selected)
                {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }
        mSettings->setAs<uint32_t>(
            "render/pt/rectLightSamplingMethod", currentRectlightSamplingMethodItemId);

        uint32_t maxDepth = mSettings->getAs<uint32_t>("render/pt/depth");
        ImGui::SliderInt("Max Depth", (int*)&maxDepth, 1, 16);
        mSettings->setAs<uint32_t>("render/pt/depth", maxDepth);

        uint32_t sppTotal = mSettings->getAs<uint32_t>("render/pt/sppTotal");
        ImGui::SliderInt("SPP Total", (int*)&sppTotal, 1, 10000);
        mSettings->setAs<uint32_t>("render/pt/sppTotal", sppTotal);

        uint32_t sppSubframe = mSettings->getAs<uint32_t>("render/pt/spp");
        ImGui::SliderInt("SPP Subframe", (int*)&sppSubframe, 1, 32);
        mSettings->setAs<uint32_t>("render/pt/spp", sppSubframe);

        bool enableAccumulation = mSettings->getAs<bool>("render/pt/enableAcc");
        ImGui::Checkbox("Enable Path Tracer Acc", &enableAccumulation);
        mSettings->setAs<bool>("render/pt/enableAcc", enableAccumulation);

        ImGui::TreePop();
    }

    if (ImGui::Button("Capture Screen"))
    {
        mSettings->setAs<bool>("render/pt/needScreenshot", true);
    }

    float cameraSpeed = mSettings->getAs<float>("render/cameraSpeed");
    ImGui::InputFloat("Camera Speed", (float*)&cameraSpeed, 0.5);
    mSettings->setAs<float>("render/cameraSpeed", cameraSpeed);
    
    const char* tonemapItems[] = { "None", "Reinhard", "ACES", "Filmic" };
    static int currentTonemapItemId = 1;
    if (ImGui::BeginCombo("Tonemap", tonemapItems[currentTonemapItemId]))
    {
        for (int n = 0; n < IM_ARRAYSIZE(tonemapItems); n++)
        {
            bool is_selected = (currentTonemapItemId == n);
            if (ImGui::Selectable(tonemapItems[n], is_selected))
            {
                currentTonemapItemId = n;
            }
            if (is_selected)
            {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }
    mSettings->setAs<uint32_t>("render/pt/tonemapperType", currentTonemapItemId);
    
    float gamma = mSettings->getAs<float>("render/post/gamma");
    ImGui::InputFloat("Gamma", (float*)&gamma, 0.5);
    mSettings->setAs<float>("render/post/gamma", gamma);

    float materialRayTmin = mSettings->getAs<float>("render/pt/dev/materialRayTmin");
    ImGui::InputFloat("Material ray T min", (float*)&materialRayTmin, 0.1);
    mSettings->setAs<float>("render/pt/dev/materialRayTmin", materialRayTmin);   
    float shadowRayTmin = mSettings->getAs<float>("render/pt/dev/shadowRayTmin");
    ImGui::InputFloat("Shadow ray T min", (float*)&shadowRayTmin, 0.1);
    mSettings->setAs<float>("render/pt/dev/shadowRayTmin", shadowRayTmin);

    ImGui::End(); // end window

    // Rendering
    ImGui::Render();
}
