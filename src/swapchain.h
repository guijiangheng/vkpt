#pragma once

#include <memory>
#include <string>
#include <vector>

#include "device.h"

namespace vkpt {

class SwapChain {
 public:
  SwapChain(Device &device, VkExtent2D windowExtent);
  SwapChain(Device &device, VkExtent2D windowExtent,
            std::shared_ptr<SwapChain> previous);

  ~SwapChain();

  VkFramebuffer getFrameBuffer(int index) { return framebuffers[index]; }

  VkExtent2D getExtent() { return swapChainExtent; }
  VkImageView getImageView(int index) { return swapChainImageViews[index]; }
  VkSwapchainKHR getSwapChain() { return swapChain; }
  VkRenderPass getRenderPass() { return renderPass; }

  float getAspectRatio() {
    return static_cast<float>(swapChainExtent.width) /
           static_cast<float>(swapChainExtent.height);
  }

  VkFormat findDepthFormat();

  VkResult acquireNextImage(uint32_t *imageIndex);

 private:
  void init();
  void createSwapChain();
  void createCommandBuffers();
  void createImageViews();
  void createColorResources();
  void createDepthResources();
  void createRenderPass();
  void createFramebuffers();

  VkSurfaceFormatKHR chooseSurfaceFormat(
      const std::vector<VkSurfaceFormatKHR> &availableFormats);
  VkPresentModeKHR choosePresentMode(
      const std::vector<VkPresentModeKHR> &availablePresentModes);
  VkExtent2D chooseExtent(const VkSurfaceCapabilitiesKHR &capabilities);

  Device &device;
  VkExtent2D windowExtent;

  VkFormat swapChainImageFormat;
  VkExtent2D swapChainExtent;

  std::vector<VkFramebuffer> framebuffers;
  VkRenderPass renderPass;

  std::vector<VkImage> swapChainImages;
  std::vector<VkImageView> swapChainImageViews;

  VkImage colorImage;
  VkDeviceMemory colorImageMemory;
  VkImageView colorImageView;

  VkImage depthImage;
  VkDeviceMemory depthImageMemory;
  VkImageView depthImageView;

  VkSwapchainKHR swapChain;
  std::shared_ptr<SwapChain> oldSwapChain;
};

}  // namespace vkpt
