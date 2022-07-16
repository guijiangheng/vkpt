#pragma once

#include <memory>
#include <string>

#include "device.h"

namespace vkpt {

class Texture {
 public:
  Texture(Device &device, std::string filepath);
  ~Texture();

  int getWidth() { return width; }
  int getHeight() { return height; }

  VkDescriptorImageInfo getImageInfo() {
    return {.sampler = sampler,
            .imageView = imageView,
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
  }

 private:
  void createImage(std::string filepath);
  void createImageView();
  void createSampler();

  Device &device;
  VkImage image{nullptr};
  VkDeviceMemory imageMemory{nullptr};
  VkImageView imageView{nullptr};
  VkSampler sampler{nullptr};
  int width{0};
  int height{0};
};

}  // namespace vkpt
