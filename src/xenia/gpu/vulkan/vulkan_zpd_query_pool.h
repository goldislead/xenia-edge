/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_VULKAN_VULKAN_ZPD_QUERY_POOL_H_
#define XENIA_GPU_VULKAN_VULKAN_ZPD_QUERY_POOL_H_

#include <cstdint>
#include <vector>

#include "xenia/base/bit_map.h"
#include "xenia/ui/vulkan/vulkan_api.h"

namespace xe {
namespace ui {
namespace vulkan {
class VulkanDevice;
}  // namespace vulkan
}  // namespace ui

namespace gpu {
namespace vulkan {

class DeferredCommandBuffer;

class VulkanZPDQueryPool {
 public:
  // Small Vulkan query pool for the ZPD path.
  //
  // Keeps query allocation, batched resolves, and readback in one place. The
  // command processor just opens and closes queries, then checks results later.
  // Vulkan can reuse indices across submissions, so this also tracks a
  // generation per slot and handles readback invalidation when needed.
  VulkanZPDQueryPool() = default;
  VulkanZPDQueryPool(const VulkanZPDQueryPool&) = delete;
  VulkanZPDQueryPool& operator=(const VulkanZPDQueryPool&) = delete;
  ~VulkanZPDQueryPool() { Shutdown(); }

  // Lifetime.
  bool EnsureInitialized(const ui::vulkan::VulkanDevice* vulkan_device,
                         uint32_t requested_capacity, bool can_recreate);
  void Shutdown();

  bool is_initialized() const {
    return query_pool_ != VK_NULL_HANDLE &&
           readback_buffer_ != VK_NULL_HANDLE && readback_mapping_ != nullptr &&
           capacity_ != 0;
  }

  uint32_t capacity() const { return capacity_; }
  bool has_pending_resolve_batch() const {
    return resolve_batch_index_count_ != 0;
  }
  bool has_free_indices() const { return !free_indices_.empty(); }

  // Allocation.
  bool AcquireQueryIndex(uint32_t& query_index, uint32_t& query_generation);
  void ReleaseQueryIndex(uint32_t query_index);
  bool GenerationMatches(uint32_t query_index, uint32_t query_generation) const;

  // Recording and resolve batching.
  void BeginQuery(DeferredCommandBuffer& deferred_command_buffer,
                  uint32_t query_index) const;
  void EndQuery(DeferredCommandBuffer& deferred_command_buffer,
                uint32_t query_index) const;

  void QueueQueryResolve(uint32_t query_index);
  void RecordResolveBatch(VkCommandBuffer command_buffer);

  // Readback.
  void InvalidateReadback();
  uint64_t GetQueryReadbackValue(uint32_t query_index) const;

 private:
  const ui::vulkan::VulkanDevice* vulkan_device_ = nullptr;

  VkQueryPool query_pool_ = VK_NULL_HANDLE;

  // Vulkan copies query results into host-visible transfer memory.
  VkBuffer readback_buffer_ = VK_NULL_HANDLE;
  VkDeviceMemory readback_memory_ = VK_NULL_HANDLE;
  uint64_t* readback_mapping_ = nullptr;
  bool readback_is_coherent_ = true;

  uint32_t capacity_ = 0;
  std::vector<uint32_t> free_indices_;

  // Vulkan can reuse query indices before older work has fully drained, so
  // each slot keeps a generation.
  std::vector<uint32_t> index_generations_;

  // Queued resolves are tracked with a bitmap and recorded in batches.
  xe::BitMap resolve_batch_index_map_;
  uint32_t resolve_batch_index_count_ = 0;
};

}  // namespace vulkan
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_VULKAN_VULKAN_ZPD_QUERY_POOL_H_
