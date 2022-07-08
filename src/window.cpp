#include "window.h"

namespace vkpt {

Window::Window(int width, int height, std::string title)
    : width{width}, height{height}, title{title} {
  glfwInit();
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
  window = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
  glfwSetWindowUserPointer(window, this);
  glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
}

Window::~Window() {
  glfwDestroyWindow(window);
  glfwTerminate();
}

void Window::framebufferResizeCallback(GLFWwindow* window, int width,
                                       int height) {
  auto win = reinterpret_cast<Window*>(glfwGetWindowUserPointer(window));
  win->framebufferResized = true;
  win->width = width;
  win->height = height;
}

}  // namespace vkpt
