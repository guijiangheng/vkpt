#include "renderer.h"

#include <array>
#include <cassert>
#include <stdexcept>

namespace vkpt {

Renderer::Renderer(Window& window, Device& device)
    : window{window}, device{device} {
  createSwapChain();
  createCommandBuffer();
}

Renderer::~Renderer() {
  freeSyncObjects();
  freeCommandBuffer();
}

void Renderer::createSwapChain() {
  auto extent = window.getFramebufferSize();
  while (extent.width == 0 || extent.height == 0) {
    extent = window.getFramebufferSize();
    glfwWaitEvents();
  }

  vkDeviceWaitIdle(device.getDevice());

  if (swapChain == nullptr) {
    swapChain = std::make_unique<SwapChain>(device, extent);
  } else {
    freeSyncObjects();
    std::shared_ptr<SwapChain> oldSwapChain = std::move(swapChain);
    swapChain = std::make_unique<SwapChain>(device, extent, oldSwapChain);
  }

  createSyncObjects();
}

void Renderer::createCommandBuffer() {
  VkCommandBufferAllocateInfo allocInfo{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = device.getCommandPool(),
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1};

  if (vkAllocateCommandBuffers(device.getDevice(), &allocInfo,
                               &commandBuffer) != VK_SUCCESS) {
    throw std::runtime_error("failed to allocate command buffers!");
  }
}

void Renderer::createSyncObjects() {
  VkSemaphoreCreateInfo semaphoreInfo = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  VkFenceCreateInfo fenceInfo = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                                 .flags = VK_FENCE_CREATE_SIGNALED_BIT};

  if (vkCreateSemaphore(device.getDevice(), &semaphoreInfo, nullptr,
                        &imageAvailableSemaphore) != VK_SUCCESS ||
      vkCreateSemaphore(device.getDevice(), &semaphoreInfo, nullptr,
                        &renderFinishedSemaphore) != VK_SUCCESS ||
      vkCreateFence(device.getDevice(), &fenceInfo, nullptr, &inFlightFence) !=
          VK_SUCCESS) {
    throw std::runtime_error(
        "failed to create synchronization objects for a frame!");
  }
}

void Renderer::freeCommandBuffer() {
  vkFreeCommandBuffers(device.getDevice(), device.getCommandPool(), 1,
                       &commandBuffer);
}

void Renderer::freeSyncObjects() {
  vkDestroySemaphore(device.getDevice(), renderFinishedSemaphore, nullptr);
  vkDestroySemaphore(device.getDevice(), imageAvailableSemaphore, nullptr);
  vkDestroyFence(device.getDevice(), inFlightFence, nullptr);
}

VkCommandBuffer Renderer::beginFrame() {
  assert(!isFrameStarted && "Can't call beginFrame while already in progress");

  vkWaitForFences(device.getDevice(), 1, &inFlightFence, VK_TRUE, UINT64_MAX);

  auto result = vkAcquireNextImageKHR(
      device.getDevice(), swapChain->getSwapChain(), UINT64_MAX,
      imageAvailableSemaphore, VK_NULL_HANDLE, &currentImageIndex);

  if (result == VK_ERROR_OUT_OF_DATE_KHR) {
    createSwapChain();
    return nullptr;
  }

  if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
    throw std::runtime_error("failed to acquire swap chain image!");
  }

  isFrameStarted = true;

  VkCommandBufferBeginInfo beginInfo{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};

  vkResetCommandBuffer(commandBuffer, 0);

  if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
    throw std::runtime_error("failed to begin recording command buffer!");
  }

  return commandBuffer;
}

void Renderer::endFrame() {
  assert(isFrameStarted &&
         "Can't call endFrame while frame is not in progress");

  if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
    throw std::runtime_error("failed to record command buffer!");
  }

  vkResetFences(device.getDevice(), 1, &inFlightFence);

  VkPipelineStageFlags waitStageMask[] = {
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
  VkSubmitInfo submitInfo{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                          .waitSemaphoreCount = 1,
                          .pWaitSemaphores = &imageAvailableSemaphore,
                          .pWaitDstStageMask = waitStageMask,
                          .commandBufferCount = 1,
                          .pCommandBuffers = &commandBuffer,
                          .signalSemaphoreCount = 1,
                          .pSignalSemaphores = &renderFinishedSemaphore};

  if (vkQueueSubmit(device.getGraphicsQueue(), 1, &submitInfo, inFlightFence) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to submit draw command buffer!");
  }

  VkSwapchainKHR swapChains[] = {swapChain->getSwapChain()};
  VkPresentInfoKHR presentInfo{.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                               .waitSemaphoreCount = 1,
                               .pWaitSemaphores = &renderFinishedSemaphore,
                               .swapchainCount = 1,
                               .pSwapchains = swapChains,
                               .pImageIndices = &currentImageIndex};

  auto result = vkQueuePresentKHR(device.getPresentQueue(), &presentInfo);

  if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR ||
      window.isFramebufferResized()) {
    window.resetResizedFlag();
    createSwapChain();
  } else if (result != VK_SUCCESS) {
    throw std::runtime_error("failed to present swap chain image!");
  }

  isFrameStarted = false;
}

void Renderer::beginRenderPass(VkCommandBuffer commandBuffer) {
  assert(isFrameStarted &&
         "Can't call beginSwapChainRenderPass if frame is not in progress");

  std::array<VkClearValue, 2> clearValues{};
  clearValues[0].color = {0.01f, 0.01f, 0.01f, 1.0f};
  clearValues[1].depthStencil = {1.0f, 0};

  VkRenderPassBeginInfo renderPassInfo{
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .renderPass = swapChain->getRenderPass(),
      .framebuffer = swapChain->getFrameBuffer(currentImageIndex),
      .renderArea = {.offset = {0, 0}, .extent = swapChain->getExtent()},
      .clearValueCount = static_cast<uint32_t>(clearValues.size()),
      .pClearValues = clearValues.data()};

  vkCmdBeginRenderPass(commandBuffer, &renderPassInfo,
                       VK_SUBPASS_CONTENTS_INLINE);

  auto extent = swapChain->getExtent();
  VkViewport viewport{.x = 0.0f,
                      .y = 0.0f,
                      .width = static_cast<float>(extent.width),
                      .height = static_cast<float>(extent.height),
                      .minDepth = 0.0f,
                      .maxDepth = 1.0f};
  VkRect2D scissor{{0, 0}, extent};
  vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
  vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
}

void Renderer::endRenderPass(VkCommandBuffer commandBuffer) {
  assert(isFrameStarted &&
         "Can't call endSwapChainRenderPass if frame is not in progress");

  vkCmdEndRenderPass(commandBuffer);
}

}  // namespace vkpt
