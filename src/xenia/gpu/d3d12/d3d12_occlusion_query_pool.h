/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_D3D12_D3D12_OCCLUSION_QUERY_POOL_H_
#define XENIA_GPU_D3D12_D3D12_OCCLUSION_QUERY_POOL_H_

#include <cstdint>
#include <vector>

#include "xenia/gpu/command_processor.h"
#include "xenia/ui/d3d12/d3d12_api.h"

namespace xe {
namespace ui {
namespace d3d12 {
class D3D12Provider;
}
}  // namespace ui

namespace gpu {
namespace d3d12 {

class DeferredCommandList;

// D3D12 occlusion query pool, used by ZPD and VIZ through separate instances.
// Results resolve into a persistent readback buffer.
//
// BeginQuery and EndQuery must be in the same command list, so callers split
// segments at EndSubmission. Discarded queries still get an EndQuery before the
// slot is released.
//
// FlushResolveBatch coalesces pending indices into contiguous ranges to cut
// down on ResolveQueryData call count.
//
// ROV queries use a separate path instead of normal D3D12 results. They write
// surviving MSAA coverage into a dedicated buffer, one slot per active query.
// QueueQueryResolve + ClearROVCounter are used instead of BeginQuery and
// EndQuery.
class D3D12OcclusionQueryPool {
 public:
  // OCCLUSION for ZPD, BINARY_OCCLUSION for VIZ.
  explicit D3D12OcclusionQueryPool(D3D12_QUERY_TYPE query_type)
      : query_type_(query_type) {}
  D3D12OcclusionQueryPool(const D3D12OcclusionQueryPool&) = delete;
  D3D12OcclusionQueryPool& operator=(const D3D12OcclusionQueryPool&) = delete;
  ~D3D12OcclusionQueryPool() { Shutdown(); }

  bool EnsureInitialized(const ui::d3d12::D3D12Provider& provider,
                         uint32_t requested_capacity, bool can_recreate,
                         bool initialize_rov_counter = false);
  bool EnsurePredicateBuffer(const ui::d3d12::D3D12Provider& provider,
                             uint32_t requested_capacity);
  void Shutdown();

  bool is_initialized() const {
    return query_heap_ && readback_buffer_ && readback_mapping_ != nullptr &&
           capacity_ != 0;
  }

  uint32_t capacity() const { return capacity_; }

  bool has_pending_resolve_batch() const {
    return !resolve_batch_indices_.empty() ||
           !rov_counter_resolve_batch_indices_.empty();
  }

  bool rov_counter_initialized() const {
    return rov_counter_buffer_ && rov_counter_readback_buffer_ &&
           rov_counter_readback_mapping_ != nullptr && capacity_ != 0;
  }

  ID3D12Resource* rov_counter_buffer() const {
    return rov_counter_buffer_.Get();
  }
  ID3D12Resource* predicate_buffer() const { return predicate_buffer_.Get(); }

  bool has_free_indices() const { return !free_indices_.empty(); }

  bool AcquireQueryIndex(uint32_t& query_index, uint32_t& query_generation);
  void ReleaseQueryIndex(uint32_t query_index, uint32_t query_generation);
  bool GenerationMatches(uint32_t query_index, uint32_t query_generation) const;

  void BeginQuery(DeferredCommandList& deferred_command_list,
                  uint32_t query_index) const;
  void EndQuery(DeferredCommandList& deferred_command_list,
                uint32_t query_index) const;
  void QueueQueryResolve(uint32_t query_index, bool uses_rov_counter = false);
  void ResolveQueryToPredicateBuffer(DeferredCommandList& deferred_command_list,
                                     uint64_t submission, uint32_t query_index,
                                     uint32_t predicate_index);
  void TransitionPredicateBuffer(DeferredCommandList& deferred_command_list,
                                 uint64_t submission,
                                 D3D12_RESOURCE_STATES new_state);
  void ClearROVCounter(DeferredCommandList& deferred_command_list,
                       uint64_t submission, uint32_t query_index);

  void FlushResolveBatch(DeferredCommandList& deferred_command_list,
                         uint64_t submission, bool submission_open);

  uint64_t GetQueryReadbackValue(uint32_t query_index,
                                 bool uses_rov_counter = false) const;

 private:
  // Transitions rov_counter_buffer_ to new_state through the tracked state,
  // starting from COMMON on the first use in each submission (buffers decay to
  // COMMON when the previous submission's command list finishes).
  void TransitionROVCounterBuffer(DeferredCommandList& deferred_command_list,
                                  uint64_t submission,
                                  D3D12_RESOURCE_STATES new_state);

  const D3D12_QUERY_TYPE query_type_;

  Microsoft::WRL::ComPtr<ID3D12QueryHeap> query_heap_;

  // Persistently mapped. Results readable once the fence signals.
  Microsoft::WRL::ComPtr<ID3D12Resource> readback_buffer_;
  uint64_t* readback_mapping_ = nullptr;

  Microsoft::WRL::ComPtr<ID3D12Resource> rov_counter_buffer_;
  Microsoft::WRL::ComPtr<ID3D12Resource> rov_counter_readback_buffer_;
  uint32_t* rov_counter_readback_mapping_ = nullptr;
  // rov_counter_buffer_ is written as a copy (WriteBufferImmediate needs
  // COPY_DEST), through pixel shader UAV atomics (UNORDERED_ACCESS), and read
  // out for resolve (COPY_SOURCE). The tracked state is only valid within the
  // submission recorded in rov_counter_buffer_state_submission_.
  D3D12_RESOURCE_STATES rov_counter_buffer_state_ = D3D12_RESOURCE_STATE_COMMON;
  uint64_t rov_counter_buffer_state_submission_ = UINT64_MAX;

  Microsoft::WRL::ComPtr<ID3D12Resource> predicate_buffer_;
  // predicate_buffer_ is written by ResolveQueryData (COPY_DEST) and read by
  // SetPredication (PREDICATION). The tracked state is only valid within the
  // submission recorded in predicate_buffer_state_submission_.
  D3D12_RESOURCE_STATES predicate_buffer_state_ = D3D12_RESOURCE_STATE_COMMON;
  uint64_t predicate_buffer_state_submission_ = UINT64_MAX;

  uint32_t capacity_ = 0;
  std::vector<uint32_t> free_indices_;

  // Bumped on each acquire so stale readbacks from a recycled slot get dropped.
  std::vector<uint32_t> index_generations_;

  std::vector<uint8_t> resolve_batch_pending_;
  // Active indices with resolve_batch_pending_[i] == 1, so flush iterates
  // only the active entries instead of scanning the full capacity.
  std::vector<uint32_t> resolve_batch_indices_;
  std::vector<uint8_t> rov_counter_resolve_batch_pending_;
  std::vector<uint32_t> rov_counter_resolve_batch_indices_;
  // Reusable scratch for coalesced contiguous ranges during flush.
  std::vector<ResolveRange> resolve_batch_ranges_;
};

}  // namespace d3d12
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_D3D12_D3D12_OCCLUSION_QUERY_POOL_H_
