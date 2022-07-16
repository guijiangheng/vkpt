#pragma once

#include <array>
#include <memory>
#include <vector>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "buffer.h"
#include "descriptor.h"
#include "model.h"
#include "pipeline.h"
#include "renderer.h"
#include "swapchain.h"
#include "texture.h"

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
  void loadModels();
  void mainLoop();
  void cleanup();

  void createDescriptorSetLayout();
  void createDescriptorSets();
  void createDescriptorPool();
  void createPipelineLayout();
  void createGraphicsPipeline();
  void createUniformBuffer();

  void updateUniformBuffer();
  void drawFrame();

  Window window{800, 600, "Vulkan"};
  Device device{window};
  Renderer renderer{window, device};
  std::unique_ptr<Pipeline> pipeline;

  std::unique_ptr<Model> model;
  std::unique_ptr<Buffer> uniformBuffer;
  Texture texture{device, "../textures/texture.jpg"};

  std::unique_ptr<DescriptorPool> globalDescriptorPool;
  std::unique_ptr<DescriptorSetLayout> descriptorSetLayout;
  VkDescriptorSet descriptorSet;
  VkPipelineLayout pipelineLayout;
};

}  // namespace vkpt
