#include "app.h"

#include <chrono>
#include <cstring>
#include <limits>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include "model.h"

namespace vkpt {

void Application::run() {
  initVulkan();
  mainLoop();
  cleanup();
}

void Application::initVulkan() {
  Model::Builder builder{
      .vertices = {{.position = {-0.5f, -0.5f, 0.0f}, .uv = {1.0f, 0.0f}},
                   {.position = {0.5f, -0.5f, 0.0f}, .uv = {0.0f, 0.0f}},
                   {.position = {0.5f, 0.5f, 0.0f}, .uv = {0.0f, 1.0f}},
                   {.position = {-0.5f, 0.5f, 0.0f}, .uv = {1.0f, 1.0f}}},
      .indices = {0, 1, 2, 2, 3, 0}};
  model = std::make_unique<Model>(device, builder);

  createDescriptorSetLayout();
  createPipelineLayout();
  createGraphicsPipeline();
  createUniformBuffers();
  createTextureImage();
  createTextureImageView();
  createTextureSampler();
  createDescriptorPool();
  createDescriptorSets();
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
  vkDestroySampler(device.getDevice(), textureSampler, nullptr);
  vkDestroyImageView(device.getDevice(), textureImageView, nullptr);
  vkDestroyImage(device.getDevice(), textureImage, nullptr);
  vkFreeMemory(device.getDevice(), textureImageMemory, nullptr);
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
  VkDescriptorImageInfo imageInfo{
      .sampler = textureSampler,
      .imageView = textureImageView,
      .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
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
  config.renderPass = renderer.getSwapchain()->getRenderPass();
  config.pipelineLayout = pipelineLayout;
  pipeline = std::make_unique<Pipeline>(device, "../shaders/simple.vert.spv",
                                        "../shaders/simple.frag.spv", config);
}

void Application::createUniformBuffers() {
  uniformBuffer =
      std::make_unique<Buffer>(device, sizeof(UniformBufferObject), 1,
                               VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
}

void Application::createTextureImage() {
  int texWidth, texHeight, texChannels;
  auto* pixels = stbi_load("../textures/texture.jpg", &texWidth, &texHeight,
                           &texChannels, STBI_rgb_alpha);
  VkDeviceSize imageSize = texWidth * texHeight * 4;

  if (!pixels) {
    throw std::runtime_error("failed to load texture image!");
  }

  VkBuffer stagingBuffer;
  VkDeviceMemory stagingBufferMemory;
  device.createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                      stagingBuffer, stagingBufferMemory);

  void* data;
  vkMapMemory(device.getDevice(), stagingBufferMemory, 0, imageSize, 0, &data);
  memcpy(data, pixels, static_cast<size_t>(imageSize));
  vkUnmapMemory(device.getDevice(), stagingBufferMemory);

  stbi_image_free(pixels);

  device.createImage(
      texWidth, texHeight, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL,
      VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, textureImage, textureImageMemory);

  device.transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_SRGB,
                               VK_IMAGE_LAYOUT_UNDEFINED,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
  device.copyBufferToImage(stagingBuffer, textureImage,
                           static_cast<uint32_t>(texWidth),
                           static_cast<uint32_t>(texHeight));
  device.transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_SRGB,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

  vkDestroyBuffer(device.getDevice(), stagingBuffer, nullptr);
  vkFreeMemory(device.getDevice(), stagingBufferMemory, nullptr);
}

void Application::createTextureImageView() {
  textureImageView = device.createImageView(
      textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT);
}

void Application::createTextureSampler() {
  auto& properties = device.getPhysicalDeviceProperties();

  VkSamplerCreateInfo samplerInfo{
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter = VK_FILTER_LINEAR,
      .minFilter = VK_FILTER_LINEAR,
      .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
      .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
      .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
      .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
      .anisotropyEnable = VK_TRUE,
      .maxAnisotropy = properties.limits.maxSamplerAnisotropy,
      .compareEnable = VK_FALSE,
      .compareOp = VK_COMPARE_OP_ALWAYS,
      .borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
      .unnormalizedCoordinates = VK_FALSE};

  if (vkCreateSampler(device.getDevice(), &samplerInfo, nullptr,
                      &textureSampler) != VK_SUCCESS) {
    throw std::runtime_error("failed to create texture sampler!");
  }
}

void Application::updateUniformBuffer() {
  static auto startTime = std::chrono::high_resolution_clock::now();

  auto currentTime = std::chrono::high_resolution_clock::now();
  float time = std::chrono::duration<float, std::chrono::seconds::period>(
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
