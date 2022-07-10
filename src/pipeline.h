#pragma once

#include <string>
#include <vector>

#include "device.h"

namespace vkpt {

struct PipelineConfig {
  std::vector<VkVertexInputBindingDescription> bindingDescriptions;
  std::vector<VkVertexInputAttributeDescription> attributeDescriptions;

  VkPipelineViewportStateCreateInfo viewportInfo;
  VkPipelineInputAssemblyStateCreateInfo inputAssemblyInfo;
  VkPipelineRasterizationStateCreateInfo rasterizationInfo;
  VkPipelineMultisampleStateCreateInfo multisampleInfo;
  VkPipelineColorBlendAttachmentState colorBlendAttachment;
  VkPipelineColorBlendStateCreateInfo colorBlendInfo;
  VkPipelineDepthStencilStateCreateInfo depthStencilInfo;
  std::vector<VkDynamicState> dynamicStates;
  VkPipelineDynamicStateCreateInfo dynamicStateInfo;

  VkPipelineLayout pipelineLayout;
  VkRenderPass renderPass;
  uint32_t subpass;
};

class Pipeline {
 public:
  Pipeline(Device& device, std::string vertFilepath, std::string fragFilepath,
           const PipelineConfig& config);
  ~Pipeline();

  void bind(VkCommandBuffer commandBuffer);

  static void populateDefaultPipelineConfig(PipelineConfig& config);
  static void enableAlphaBlending(PipelineConfig& config);

 private:
  void createGraphicsPipeline(std::string vertFilepath,
                              std::string fragFilepath,
                              const PipelineConfig& config);

  VkShaderModule createShaderModule(const std::vector<char>& code);

  Device& device;
  VkPipeline graphicsPipeline;
};
}  // namespace vkpt
