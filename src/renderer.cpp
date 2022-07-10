#include "renderer.h"

#include <array>
#include <cassert>
#include <stdexcept>

namespace vkpt {

Renderer::Renderer(Window& window, Device& device)
    : window{window}, device{device} {
  createSwapChain();
  createCommandBuffers();
}

Renderer::~Renderer() {
  freeSyncObjects();
  freeCommandBuffers();
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

void Renderer::createCommandBuffers() {
  VkCommandBufferAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandPool = device.getCommandPool();
  allocInfo.commandBufferCount = static_cast<uint32_t>(commandBuffers.size());

  if (vkAllocateCommandBuffers(device.getDevice(), &allocInfo,
                               commandBuffers.data()) != VK_SUCCESS) {
    throw std::runtime_error("failed to allocate command buffers!");
  }
}

void Renderer::createSyncObjects() {
  VkSemaphoreCreateInfo semaphoreInfo = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  VkFenceCreateInfo fenceInfo = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                                 .flags = VK_FENCE_CREATE_SIGNALED_BIT};

  for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    if (vkCreateSemaphore(device.getDevice(), &semaphoreInfo, nullptr,
                          &imageAvailableSemaphores[i]) != VK_SUCCESS ||
        vkCreateSemaphore(device.getDevice(), &semaphoreInfo, nullptr,
                          &renderFinishedSemaphores[i]) != VK_SUCCESS ||
        vkCreateFence(device.getDevice(), &fenceInfo, nullptr,
                      &inFlightFences[i]) != VK_SUCCESS) {
      throw std::runtime_error(
          "failed to create synchronization objects for a frame!");
    }
  }
}

void Renderer::freeCommandBuffers() {
  vkFreeCommandBuffers(device.getDevice(), device.getCommandPool(),
                       static_cast<uint32_t>(commandBuffers.size()),
                       commandBuffers.data());
}

void Renderer::freeSyncObjects() {
  for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    vkDestroySemaphore(device.getDevice(), renderFinishedSemaphores[i],
                       nullptr);
    vkDestroySemaphore(device.getDevice(), imageAvailableSemaphores[i],
                       nullptr);
    vkDestroyFence(device.getDevice(), inFlightFences[i], nullptr);
  }
}

VkCommandBuffer Renderer::beginFrame() {
  assert(!isFrameStarted && "Can't call beginFrame while already in progress");

  vkWaitForFences(device.getDevice(), 1, &inFlightFences[currentFrameIndex],
                  VK_TRUE, UINT64_MAX);

  auto result = vkAcquireNextImageKHR(
      device.getDevice(), swapChain->getSwapChain(), UINT64_MAX,
      imageAvailableSemaphores[currentFrameIndex], VK_NULL_HANDLE,
      &currentImageIndex);

  if (result == VK_ERROR_OUT_OF_DATE_KHR) {
    createSwapChain();
    return nullptr;
  }

  if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
    throw std::runtime_error("failed to acquire swap chain image!");
  }

  isFrameStarted = true;

  auto commandBuffer = commandBuffers[currentFrameIndex];
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

  auto commandBuffer = commandBuffers[currentFrameIndex];

  if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
    throw std::runtime_error("failed to record command buffer!");
  }

  vkResetFences(device.getDevice(), 1, &inFlightFences[currentFrameIndex]);

  VkPipelineStageFlags waitStageMask[] = {
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
  VkSubmitInfo submitInfo{
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .waitSemaphoreCount = 1,
      .pWaitSemaphores = &imageAvailableSemaphores[currentFrameIndex],
      .pWaitDstStageMask = waitStageMask,
      .commandBufferCount = 1,
      .pCommandBuffers = &commandBuffer,
      .signalSemaphoreCount = 1,
      .pSignalSemaphores = &renderFinishedSemaphores[currentFrameIndex]};

  if (vkQueueSubmit(device.getGraphicsQueue(), 1, &submitInfo,
                    inFlightFences[currentFrameIndex]) != VK_SUCCESS) {
    throw std::runtime_error("failed to submit draw command buffer!");
  }

  VkPresentInfoKHR presentInfo{
      .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
      .waitSemaphoreCount = 1,
      .pWaitSemaphores = &renderFinishedSemaphores[currentFrameIndex],
      .swapchainCount = 1,
      .pSwapchains = &swapChain->getSwapChain(),
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
  currentFrameIndex = (currentFrameIndex + 1) % MAX_FRAMES_IN_FLIGHT;
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
