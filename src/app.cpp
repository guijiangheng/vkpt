#include "app.h"

#include <chrono>
#include <cstring>
#include <limits>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "model.h"

namespace vkpt {

void Application::run() {
  loadModels();
  initVulkan();
  mainLoop();
  cleanup();
}

void Application::initVulkan() {
  createDescriptorSetLayout();
  createPipelineLayout();
  createGraphicsPipeline();
  createUniformBuffer();
  createDescriptorPool();
  createDescriptorSets();
}

void Application::loadModels() {
  model = Model::fromFile(device, "../resources/viking_room.obj");
  texture = std::make_unique<Texture>(device, "../resources/viking_room.png");
}

void Application::mainLoop() {
  while (!window.shouldClose()) {
    glfwPollEvents();
    drawFrame();
  }

  vkDeviceWaitIdle(device.getDevice());
}

void Application::cleanup() {
  vkDestroyPipelineLayout(device.getDevice(), pipelineLayout, nullptr);
}

void Application::createDescriptorSetLayout() {
  descriptorSetLayout =
      DescriptorSetLayout::Builder(device)
          .addBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                      VK_SHADER_STAGE_VERTEX_BIT)
          .addBinding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                      VK_SHADER_STAGE_FRAGMENT_BIT)
          .build();
}

void Application::createDescriptorSets() {
  auto bufferInfo =
      uniformBuffer->getDescriptorInfo(sizeof(UniformBufferObject));
  auto imageInfo = texture->getImageInfo();
  DescriptorWriter(*descriptorSetLayout, *globalDescriptorPool)
      .writeBuffer(0, &bufferInfo)
      .writeImage(1, &imageInfo)
      .build(descriptorSet);
}

void Application::createDescriptorPool() {
  globalDescriptorPool =
      DescriptorPool::Builder(device)
          .setMaxSets(1)
          .addPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1)
          .addPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1)
          .build();
}

void Application::createPipelineLayout() {
  VkDescriptorSetLayout setLayouts[] = {
      descriptorSetLayout->getDescriptorSetLayout()};
  VkPipelineLayoutCreateInfo pipelineLayoutInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = setLayouts,
      .pushConstantRangeCount = 0};

  if (vkCreatePipelineLayout(device.getDevice(), &pipelineLayoutInfo, nullptr,
                             &pipelineLayout) != VK_SUCCESS) {
    throw std::runtime_error("failed to create pipeline layout!");
  }
}

void Application::createGraphicsPipeline() {
  PipelineConfig config{};
  Pipeline::populateDefaultPipelineConfig(config);
  config.multisampleInfo.rasterizationSamples = device.getMsaaSamples();
  config.renderPass = renderer.getSwapchain()->getRenderPass();
  config.pipelineLayout = pipelineLayout;
  pipeline = std::make_unique<Pipeline>(device, "../shaders/simple.vert.spv",
                                        "../shaders/simple.frag.spv", config);
}

void Application::createUniformBuffer() {
  uniformBuffer =
      std::make_unique<Buffer>(device, sizeof(UniformBufferObject), 1,
                               VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
}

void Application::updateUniformBuffer() {
  static auto startTime = std::chrono::high_resolution_clock::now();

  auto currentTime = std::chrono::high_resolution_clock::now();
  auto time = std::chrono::duration<float, std::chrono::seconds::period>(
                  currentTime - startTime)
                  .count();

  auto extent = renderer.getSwapchain()->getExtent();
  UniformBufferObject ubo{
      .model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f),
                           glm::vec3(0.0f, 0.0f, 1.0f)),
      .view =
          glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f),
                      glm::vec3(0.0f, 0.0f, 1.0f)),
      .proj = glm::perspective(glm::radians(45.0f),
                               (float)extent.width / (float)extent.height, 0.1f,
                               10.0f)};

  ubo.proj[1][1] *= -1;

  uniformBuffer->map();
  uniformBuffer->writeToBuffer((void*)&ubo);
  uniformBuffer->unmap();
}

void Application::drawFrame() {
  if (auto commandBuffer = renderer.beginFrame()) {
    updateUniformBuffer();
    renderer.beginRenderPass(commandBuffer);
    pipeline->bind(commandBuffer);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
    model->bind(commandBuffer);
    model->draw(commandBuffer);
    renderer.endRenderPass(commandBuffer);
    renderer.endFrame();
  }
}

}  // namespace vkpt
