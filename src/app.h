#pragma once

#define GLM_FORCE_RADIANS
#include <array>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <memory>
#include <vector>

#include "buffer.h"

namespace vkpt {

constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2;

struct UniformBufferObject {
  glm::mat4 model;
  glm::mat4 view;
  glm::mat4 proj;
};

struct Vertex {
  glm::vec2 pos;
  glm::vec3 color;
  glm::vec2 texCoord;

  static VkVertexInputBindingDescription getBindingDescription() {
    return VkVertexInputBindingDescription{
        .binding = 0,
        .stride = sizeof(Vertex),
        .inputRate = VK_VERTEX_INPUT_RATE_VERTEX};
  }

  static std::array<VkVertexInputAttributeDescription, 3>
  getAttributeDescriptions() {
    return {
        VkVertexInputAttributeDescription{.location = 0,
                                          .binding = 0,
                                          .format = VK_FORMAT_R32G32_SFLOAT,
                                          .offset = offsetof(Vertex, pos)},
        VkVertexInputAttributeDescription{.location = 1,
                                          .binding = 0,
                                          .format = VK_FORMAT_R32G32B32_SFLOAT,
                                          .offset = offsetof(Vertex, color)},
        VkVertexInputAttributeDescription{
            .location = 2,
            .binding = 0,
            .format = VK_FORMAT_R32G32_SFLOAT,
            .offset = offsetof(Vertex, texCoord)}};
  }
};

class Application {
 public:
  void run();

 private:
  Window window{800, 600, "Vulkan"};
  Device device{window};

  VkSwapchainKHR swapChain;
  std::vector<VkImage> swapChainImages;
  VkFormat swapChainImageFormat;
  VkExtent2D swapChainExtent;
  std::vector<VkImageView> swapChainImageViews;
  std::vector<VkFramebuffer> swapChainFramebuffers;

  VkRenderPass renderPass;
  VkDescriptorSetLayout descriptorSetLayout;
  VkPipelineLayout pipelineLayout;
  VkPipeline graphicsPipeline;

  std::vector<VkCommandBuffer> commandBuffers;

  VkDescriptorPool descriptorPool;
  std::vector<VkDescriptorSet> descriptorSets;

  std::unique_ptr<Buffer> vertexBuffer;
  std::unique_ptr<Buffer> indexBuffer;
  std::vector<std::unique_ptr<Buffer>> uniformBuffers;

  VkImage textureImage;
  VkDeviceMemory textureImageMemory;
  VkImageView textureImageView;
  VkSampler textureSampler;

  std::vector<VkSemaphore> imageAvailableSemaphores;
  std::vector<VkSemaphore> renderFinishedSemaphores;
  std::vector<VkFence> inFlightFences;
  uint32_t currentFrame = 0;

  void initVulkan();
  void mainLoop();
  void cleanup();

  void createSwapChain();
  void createImageViews();
  void createRenderPass();
  void createDescriptorSetLayout();
  void createGraphicsPipeline();
  void createFramebuffers();
  void createCommandBuffers();
  void createSyncObjects();
  void createVertexBuffer();
  void createIndexBuffer();
  void createUniformBuffers();
  void createTextureImage();
  void createTextureImageView();
  void createTextureSampler();
  void createDescriptorPool();
  void createDescriptorSets();

  void cleanupSwapChain();
  void recreateSwapChain();

  void updateUniformBuffer(uint32_t currentImage);
  void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex);
  void drawFrame();

  VkPresentModeKHR choosePresentMode(
      const std::vector<VkPresentModeKHR>& modes);
  VkSurfaceFormatKHR chooseSurfaceFormat(
      const std::vector<VkSurfaceFormatKHR>& formats);
  VkExtent2D chooseExtent(const VkSurfaceCapabilitiesKHR& capabilities);

  VkShaderModule createShaderModule(const std::vector<char>& code);
};

}  // namespace vkpt
