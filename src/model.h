#pragma once

#include <vulkan/vulkan.h>

#include <vector>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace vkpt {

struct Vertex {
  glm::vec2 pos;
  glm::vec3 color;
  glm::vec2 texCoord;

  static std::vector<VkVertexInputBindingDescription> getBindingDescriptions() {
    return {{.binding = 0,
             .stride = sizeof(Vertex),
             .inputRate = VK_VERTEX_INPUT_RATE_VERTEX}};
  }

  static std::vector<VkVertexInputAttributeDescription>
  getAttributeDescriptions() {
    return {{.location = 0,
             .binding = 0,
             .format = VK_FORMAT_R32G32_SFLOAT,
             .offset = offsetof(Vertex, pos)},
            {.location = 1,
             .binding = 0,
             .format = VK_FORMAT_R32G32B32_SFLOAT,
             .offset = offsetof(Vertex, color)},
            {.location = 2,
             .binding = 0,
             .format = VK_FORMAT_R32G32_SFLOAT,
             .offset = offsetof(Vertex, texCoord)}};
  }
};

}  // namespace  vkpt
