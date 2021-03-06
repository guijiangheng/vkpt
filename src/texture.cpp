#include "texture.h"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

namespace vkpt {

Texture::Texture(Device& device, std::string filepath) : device(device) {
  createImage(filepath);
  createImageView();
  createSampler();
}

Texture::~Texture() {
  vkDestroySampler(device.getDevice(), sampler, nullptr);
  vkDestroyImageView(device.getDevice(), imageView, nullptr);
  vkDestroyImage(device.getDevice(), image, nullptr);
  vkFreeMemory(device.getDevice(), imageMemory, nullptr);
}

void Texture::createImage(std::string filepath) {
  int w, h, texChannels;
  auto* pixels =
      stbi_load(filepath.c_str(), &w, &h, &texChannels, STBI_rgb_alpha);
  width = static_cast<uint32_t>(w);
  height = static_cast<uint32_t>(h);
  mipLevels = getMipLevels(width, height);
  VkDeviceSize imageSize = width * height * 4;

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

  device.createImage(width, height, mipLevels, VK_SAMPLE_COUNT_1_BIT,
                     VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL,
                     VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                         VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                         VK_IMAGE_USAGE_SAMPLED_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, image, imageMemory);

  device.transitionImageLayout(image, VK_FORMAT_R8G8B8A8_SRGB,
                               VK_IMAGE_LAYOUT_UNDEFINED,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, mipLevels);
  device.copyBufferToImage(stagingBuffer, image, static_cast<uint32_t>(width),
                           static_cast<uint32_t>(height));
  vkDestroyBuffer(device.getDevice(), stagingBuffer, nullptr);
  vkFreeMemory(device.getDevice(), stagingBufferMemory, nullptr);
  device.generateMipmaps(image, VK_FORMAT_R8G8B8A8_SRGB, width, height,
                         mipLevels);
}

void Texture::createImageView() {
  imageView = device.createImageView(image, VK_FORMAT_R8G8B8A8_SRGB,
                                     VK_IMAGE_ASPECT_COLOR_BIT, mipLevels);
}

void Texture::createSampler() {
  auto& properties = device.getPhysicalDeviceProperties();

  VkSamplerCreateInfo samplerInfo{
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter = VK_FILTER_LINEAR,
      .minFilter = VK_FILTER_LINEAR,
      .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
      .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
      .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
      .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
      .mipLodBias = 0.0f,
      .anisotropyEnable = VK_TRUE,
      .maxAnisotropy = properties.limits.maxSamplerAnisotropy,
      .compareEnable = VK_FALSE,
      .compareOp = VK_COMPARE_OP_ALWAYS,
      .minLod = 0.0f,
      .maxLod = static_cast<float>(mipLevels),
      .borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
      .unnormalizedCoordinates = VK_FALSE};

  if (vkCreateSampler(device.getDevice(), &samplerInfo, nullptr, &sampler) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to create texture sampler!");
  }
}

}  // namespace vkpt
