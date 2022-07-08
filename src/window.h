#pragma once

#include <stdexcept>
#include <string>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

namespace vkpt {

class Window {
 public:
  Window(int width, int height, std::string title);
  ~Window();

  bool shouldClose() const { return glfwWindowShouldClose(window); }

  bool getFramebufferResized() const { return framebufferResized; }

  void resetResizedFlag() { framebufferResized = false; }

  void createSurface(VkInstance instance, VkSurfaceKHR& surface) const {
    if (glfwCreateWindowSurface(instance, window, nullptr, &surface) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create window surface!");
    }
  }

  VkExtent2D getExtent() const {
    return {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
  }

 private:
  static void framebufferResizeCallback(GLFWwindow* window, int width,
                                        int height);

  int width, height;
  bool framebufferResized = false;
  std::string title;
  GLFWwindow* window;
};

}  // namespace vkpt
