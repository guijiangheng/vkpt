#pragma once

#include <array>
#include <cassert>
#include <memory>

#include "swapchain.h"
#include "window.h"

namespace vkpt {
class Renderer {
 public:
  Renderer(Window &window, Device &device);
  ~Renderer();

  SwapChain *getSwapchain() { return swapChain.get(); }

  VkCommandBuffer beginFrame();
  void endFrame();
  void beginRenderPass(VkCommandBuffer commandBuffer);
  void endRenderPass(VkCommandBuffer commandBuffer);

 private:
  void createSwapChain();
  void createCommandBuffer();
  void createSyncObjects();

  void freeCommandBuffer();
  void freeSyncObjects();

  Window &window;
  Device &device;
  std::unique_ptr<SwapChain> swapChain;

  VkCommandBuffer commandBuffer;
  VkSemaphore imageAvailableSemaphore;
  VkSemaphore renderFinishedSemaphore;
  VkFence inFlightFence;

  uint32_t currentImageIndex;
  bool isFrameStarted{false};
};

}  // namespace vkpt
