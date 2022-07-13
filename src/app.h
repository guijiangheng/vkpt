#pragma once

#include <array>
#include <memory>
#include <vector>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "buffer.h"
#include "descriptor.h"
#include "pipeline.h"
#include "renderer.h"
#include "swapchain.h"

namespace vkpt {

struct UniformBufferObject {
  glm::mat4 model;
  glm::mat4 view;
  glm::mat4 proj;
};

class Application {
 public:
  void run();

 private:
  void initVulkan();
  void mainLoop();
  void cleanup();

  void createDescriptorSetLayout();
  void createDescriptorSets();
  void createDescriptorPool();
  void createPipelineLayout();
  void createGraphicsPipeline();
  void createVertexBuffer();
  void createIndexBuffer();
  void createUniformBuffers();
  void createTextureImage();
  void createTextureImageView();
  void createTextureSampler();

  void updateUniformBuffer(uint32_t currentImage);
  void drawFrame();

  Window window{800, 600, "Vulkan"};
  Device device{window};
  Renderer renderer{window, device};
  std::unique_ptr<Pipeline> pipeline;

  std::unique_ptr<Buffer> vertexBuffer;
  std::unique_ptr<Buffer> indexBuffer;
  std::vector<std::unique_ptr<Buffer>> uniformBuffers;

  std::unique_ptr<DescriptorPool> globalDescriptorPool;
  std::unique_ptr<DescriptorSetLayout> descriptorSetLayout;
  std::vector<VkDescriptorSet> descriptorSets;

  VkPipelineLayout pipelineLayout;

  VkImage textureImage;
  VkDeviceMemory textureImageMemory;
  VkImageView textureImageView;
  VkSampler textureSampler;
};

}  // namespace vkpt
