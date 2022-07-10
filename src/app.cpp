#include "app.h"

#include <chrono>
#include <cstring>
#include <limits>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include "model.h"

namespace vkpt {

const std::vector<Vertex> vertices = {
    {{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
    {{0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
    {{0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},
    {{-0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f}}};
const std::vector<uint16_t> indices = {0, 1, 2, 2, 3, 0};

void Application::run() {
  initVulkan();
  mainLoop();
  cleanup();
}

void Application::initVulkan() {
  createDescriptorSetLayout();
  createPipelineLayout();
  createGraphicsPipeline();
  createVertexBuffer();
  createIndexBuffer();
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

  vkDestroyDescriptorPool(device.getDevice(), descriptorPool, nullptr);
  vkDestroyDescriptorSetLayout(device.getDevice(), descriptorSetLayout,
                               nullptr);

  vkDestroySampler(device.getDevice(), textureSampler, nullptr);
  vkDestroyImageView(device.getDevice(), textureImageView, nullptr);
  vkDestroyImage(device.getDevice(), textureImage, nullptr);
  vkFreeMemory(device.getDevice(), textureImageMemory, nullptr);
}

void Application::createDescriptorSetLayout() {
  std::vector<VkDescriptorSetLayoutBinding> bindings = {
      {.binding = 0,
       .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
       .descriptorCount = 1,
       .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
       .pImmutableSamplers = nullptr},

      {.binding = 1,
       .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
       .descriptorCount = 1,
       .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
       .pImmutableSamplers = nullptr}};

  VkDescriptorSetLayoutCreateInfo layoutInfo{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = static_cast<uint32_t>(bindings.size()),
      .pBindings = bindings.data()};

  if (vkCreateDescriptorSetLayout(device.getDevice(), &layoutInfo, nullptr,
                                  &descriptorSetLayout) != VK_SUCCESS) {
    throw std::runtime_error("failed to create descriptor set layout!");
  }
}

void Application::createDescriptorSets() {
  std::vector<VkDescriptorSetLayout> layouts(Renderer::MAX_FRAMES_IN_FLIGHT,
                                             descriptorSetLayout);
  VkDescriptorSetAllocateInfo allocInfo{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = descriptorPool,
      .descriptorSetCount = Renderer::MAX_FRAMES_IN_FLIGHT,
      .pSetLayouts = layouts.data()};

  descriptorSets.resize(Renderer::MAX_FRAMES_IN_FLIGHT);

  if (vkAllocateDescriptorSets(device.getDevice(), &allocInfo,
                               descriptorSets.data()) != VK_SUCCESS) {
    throw std::runtime_error("failed to allocate descriptor sets!");
  }

  for (size_t i = 0; i < Renderer::MAX_FRAMES_IN_FLIGHT; i++) {
    auto bufferInfo =
        uniformBuffers[i]->getDescriptorInfo(sizeof(UniformBufferObject));

    VkDescriptorImageInfo imageInfo{
        .sampler = textureSampler,
        .imageView = textureImageView,
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};

    std::vector<VkWriteDescriptorSet> descriptorWrites = {
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = descriptorSets[i],
         .dstBinding = 0,
         .dstArrayElement = 0,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
         .pBufferInfo = &bufferInfo},

        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = descriptorSets[i],
         .dstBinding = 1,
         .dstArrayElement = 0,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         .pImageInfo = &imageInfo},
    };

    vkUpdateDescriptorSets(device.getDevice(),
                           static_cast<uint32_t>(descriptorWrites.size()),
                           descriptorWrites.data(), 0, nullptr);
  }
}

void Application::createDescriptorPool() {
  std::vector<VkDescriptorPoolSize> poolSizes = {
      {.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
       .descriptorCount = Renderer::MAX_FRAMES_IN_FLIGHT},
      {.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
       .descriptorCount = Renderer::MAX_FRAMES_IN_FLIGHT}};

  VkDescriptorPoolCreateInfo poolInfo{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = Renderer::MAX_FRAMES_IN_FLIGHT,
      .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
      .pPoolSizes = poolSizes.data()};

  if (vkCreateDescriptorPool(device.getDevice(), &poolInfo, nullptr,
                             &descriptorPool) != VK_SUCCESS) {
    throw std::runtime_error("failed to create descriptor pool!");
  }
}

void Application::createPipelineLayout() {
  VkPipelineLayoutCreateInfo pipelineLayoutInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = &descriptorSetLayout,
      .pushConstantRangeCount = 0};

  if (vkCreatePipelineLayout(device.getDevice(), &pipelineLayoutInfo, nullptr,
                             &pipelineLayout) != VK_SUCCESS) {
    throw std::runtime_error("failed to create pipeline layout!");
  }
}

void Application::createGraphicsPipeline() {
  PipelineConfig config{};
  Pipeline::populateDefaultPipelineConfig(config);
  config.renderPass = renderer.getRenderPass();
  config.pipelineLayout = pipelineLayout;
  pipeline = std::make_unique<Pipeline>(device, "../shaders/simple.vert.spv",
                                        "../shaders/simple.frag.spv", config);
}

void Application::createVertexBuffer() {
  auto instanceCount = static_cast<uint32_t>(vertices.size());
  Buffer stagingBuffer{device, sizeof(vertices[0]), instanceCount,
                       VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT};
  stagingBuffer.map();
  stagingBuffer.writeToBuffer((void*)vertices.data());

  vertexBuffer = std::make_unique<Buffer>(
      device, sizeof(vertices[0]), instanceCount,
      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  device.copyBuffer(stagingBuffer.getBuffer(), vertexBuffer->getBuffer(),
                    sizeof(vertices[0]) * instanceCount);
}

void Application::createIndexBuffer() {
  auto instanceCount = static_cast<uint32_t>(indices.size());
  Buffer stagingBuffer{device, sizeof(indices[0]), instanceCount,
                       VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT};
  stagingBuffer.map();
  stagingBuffer.writeToBuffer((void*)indices.data());

  indexBuffer = std::make_unique<Buffer>(
      device, sizeof(indices[0]), instanceCount,
      VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  device.copyBuffer(stagingBuffer.getBuffer(), indexBuffer->getBuffer(),
                    sizeof(indices[0]) * instanceCount);
}

void Application::createUniformBuffers() {
  VkDeviceSize bufferSize = sizeof(UniformBufferObject);
  uniformBuffers.reserve(Renderer::MAX_FRAMES_IN_FLIGHT);

  for (uint32_t i = 0; i < Renderer::MAX_FRAMES_IN_FLIGHT; ++i) {
    uniformBuffers.push_back(std::make_unique<Buffer>(
        device, bufferSize, 1, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT));
  }
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

void Application::updateUniformBuffer(uint32_t imageIndex) {
  static auto startTime = std::chrono::high_resolution_clock::now();

  auto currentTime = std::chrono::high_resolution_clock::now();
  float time = std::chrono::duration<float, std::chrono::seconds::period>(
                   currentTime - startTime)
                   .count();

  auto extent = renderer.getSwapchainExtent();
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

  uniformBuffers[imageIndex]->map();
  uniformBuffers[imageIndex]->writeToBuffer((void*)&ubo);
  uniformBuffers[imageIndex]->unmap();
}

void Application::drawFrame() {
  if (auto commandBuffer = renderer.beginFrame()) {
    auto currentFrame = renderer.getCurrentFrame();
    updateUniformBuffer(currentFrame);

    renderer.beginRenderPass(commandBuffer);
    pipeline->bind(commandBuffer);

    VkBuffer vertexBuffers[] = {vertexBuffer->getBuffer()};
    VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
    vkCmdBindIndexBuffer(commandBuffer, indexBuffer->getBuffer(), 0,
                         VK_INDEX_TYPE_UINT16);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipelineLayout, 0, 1, &descriptorSets[currentFrame],
                            0, nullptr);
    vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(indices.size()), 1, 0,
                     0, 0);

    renderer.endRenderPass(commandBuffer);
    renderer.endFrame();
  }
}

}  // namespace vkpt
