/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_D3D12_D3D12_ZPD_QUERY_POOL_H_
#define XENIA_GPU_D3D12_D3D12_ZPD_QUERY_POOL_H_

#include <cstdint>
#include <vector>

#include "xenia/base/bit_map.h"
#include "xenia/ui/d3d12/d3d12_api.h"

namespace xe {
namespace ui {
namespace d3d12 {
class D3D12Provider;
}  // namespace d3d12
}  // namespace ui

namespace gpu {
namespace d3d12 {

class DeferredCommandList;

class D3D12ZPDQueryPool {
 public:
  // Small D3D12 query pool for the ZPD path.
  //
  // Keeps query allocation, batched resolves, and readback in one place. The
  // command processor just opens and closes queries, then reads results back
  // later. D3D12 can resolve straight into readback memory, so this stays
  // fairly small.
  D3D12ZPDQueryPool() = default;
  D3D12ZPDQueryPool(const D3D12ZPDQueryPool&) = delete;
  D3D12ZPDQueryPool& operator=(const D3D12ZPDQueryPool&) = delete;
  ~D3D12ZPDQueryPool() { Shutdown(); }

  // Lifetime.
  bool EnsureInitialized(const ui::d3d12::D3D12Provider& provider,
                         uint32_t requested_capacity, bool can_recreate);
  void Shutdown();

  bool is_initialized() const {
    return query_heap_ && readback_buffer_ && readback_mapping_ != nullptr &&
           capacity_ != 0;
  }

  uint32_t capacity() const { return capacity_; }
  bool has_pending_resolve_batch() const {
    return resolve_batch_index_count_ != 0;
  }
  bool has_free_indices() const { return !free_indices_.empty(); }

  // Allocation.
  bool AcquireQueryIndex(uint32_t& query_index,
                         uint32_t& query_generation);
  void ReleaseQueryIndex(uint32_t query_index, uint32_t query_generation);
  bool GenerationMatches(uint32_t query_index,
                         uint32_t query_generation) const;

  // Recording and resolve batching.
  void BeginQuery(DeferredCommandList& deferred_command_list,
                  uint32_t query_index) const;
  void EndQuery(DeferredCommandList& deferred_command_list,
                uint32_t query_index) const;

  void QueueQueryResolve(uint32_t query_index);
  void FlushResolveBatch(DeferredCommandList& deferred_command_list,
                         bool submission_open);

  // Readback.
  uint64_t GetQueryReadbackValue(uint32_t query_index) const;

 private:
  Microsoft::WRL::ComPtr<ID3D12QueryHeap> query_heap_;

  // D3D12 can resolve straight into readback memory, so one mapping is
  // enough here.
  Microsoft::WRL::ComPtr<ID3D12Resource> readback_buffer_;
  uint64_t* readback_mapping_ = nullptr;

  uint32_t capacity_ = 0;
  std::vector<uint32_t> free_indices_;

  // D3D12 keeps indices alive until they are explicitly released, but a
  // generation still helps catch stale or double-retired handles before an old
  // readback value can be reused accidentally.
  std::vector<uint32_t> index_generations_;

  // Queued resolves are tracked with a bitmap and flushed in batches.
  xe::BitMap resolve_batch_index_map_;
  uint32_t resolve_batch_index_count_ = 0;
};

}  // namespace d3d12
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_D3D12_D3D12_ZPD_QUERY_POOL_H_
