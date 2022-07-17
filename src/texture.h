#pragma once

#include <cmath>
#include <memory>
#include <string>

#include "device.h"

namespace vkpt {

class Texture {
 public:
  Texture(Device &device, std::string filepath);
  ~Texture();

  uint32_t getWidth() { return width; }
  uint32_t getHeight() { return height; }

  VkDescriptorImageInfo getImageInfo() {
    return {.sampler = sampler,
            .imageView = imageView,
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
  }

 private:
  static uint32_t getMipLevels(uint32_t width, uint32_t height) {
    return static_cast<uint32_t>(
               std::floor(std::log2(std::max(width, height)))) +
           1;
  }

  void createImage(std::string filepath);
  void createImageView();
  void createSampler();

  Device &device;
  VkImage image{nullptr};
  VkDeviceMemory imageMemory{nullptr};
  VkImageView imageView{nullptr};
  VkSampler sampler{nullptr};
  uint32_t width{0};
  uint32_t height{0};
  uint32_t mipLevels{0};
};

}  // namespace vkpt
