#pragma once

#include <memory>
#include <vector>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>

#include "buffer.h"
#include "device.h"

namespace vkpt {

class Model {
 public:
  struct Vertex {
    glm::vec3 position;
    glm::vec3 color;
    glm::vec3 normal;
    glm::vec2 uv;

    static std::vector<VkVertexInputBindingDescription>
    getBindingDescriptions();
    static std::vector<VkVertexInputAttributeDescription>
    getAttributeDescriptions();

    bool operator==(const Vertex &rhs) const {
      return position == rhs.position && color == rhs.color &&
             normal == rhs.normal && uv == rhs.uv;
    }
  };

  struct Builder {
    std::vector<Vertex> vertices{};
    std::vector<uint32_t> indices{};

    void loadModel(std::string filepath);
  };

  Model(Device &device, const Model::Builder &builder);

  static std::unique_ptr<Model> fromFile(Device &device, std::string filepath);

  void bind(VkCommandBuffer commandBuffer);
  void draw(VkCommandBuffer commandBuffer);

 private:
  void createVertexBuffers(const std::vector<Vertex> &vertices);
  void createIndexBuffers(const std::vector<uint32_t> &indices);

  Device &device;
  std::unique_ptr<Buffer> vertexBuffer;
  std::unique_ptr<Buffer> indexBuffer;
  uint32_t vertexCount;
  uint32_t indexCount;
  bool hasIndexBuffer = false;
};

}  // namespace vkpt
