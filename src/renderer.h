#pragma once

#include <cassert>
#include <memory>

#include "swapchain.h"
#include "window.h"

namespace vkpt {
class Renderer {
 public:
  static constexpr auto MAX_FRAMES_IN_FLIGHT = 2;

  Renderer(Window &window, Device &device);
  ~Renderer();

  int getCurrentFrame() { return currentFrameIndex; }
  VkExtent2D getSwapchainExtent() { return swapChain->getExtent(); }
  VkRenderPass getRenderPass() { return swapChain->getRenderPass(); }

  VkCommandBuffer beginFrame();
  void endFrame();
  void beginRenderPass(VkCommandBuffer commandBuffer);
  void endRenderPass(VkCommandBuffer commandBuffer);

 private:
  void createSwapChain();
  void createCommandBuffers();
  void createSyncObjects();

  void freeCommandBuffers();
  void freeSyncObjects();

  Window &window;
  Device &device;
  std::unique_ptr<SwapChain> swapChain;

  std::array<VkCommandBuffer, MAX_FRAMES_IN_FLIGHT> commandBuffers;
  std::array<VkSemaphore, MAX_FRAMES_IN_FLIGHT> imageAvailableSemaphores;
  std::array<VkSemaphore, MAX_FRAMES_IN_FLIGHT> renderFinishedSemaphores;
  std::array<VkFence, MAX_FRAMES_IN_FLIGHT> inFlightFences;

  uint32_t currentImageIndex;
  int currentFrameIndex{0};
  bool isFrameStarted{false};
};

}  // namespace vkpt
