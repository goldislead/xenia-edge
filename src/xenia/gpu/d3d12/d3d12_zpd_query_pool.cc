/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/gpu/d3d12/d3d12_zpd_query_pool.h"

#include "xenia/base/logging.h"
#include "xenia/gpu/gpu_flags.h"
#include "xenia/gpu/command_processor.h"
#include "xenia/gpu/d3d12/deferred_command_list.h"
#include "xenia/ui/d3d12/d3d12_provider.h"
#include "xenia/ui/d3d12/d3d12_util.h"

namespace xe {
namespace gpu {
namespace d3d12 {

bool D3D12ZPDQueryPool::EnsureInitialized(
    const ui::d3d12::D3D12Provider& provider, uint32_t requested_capacity,
    bool can_recreate) {
  if (is_initialized() && capacity_ == requested_capacity) {
    return true;
  }

  if (is_initialized() && !can_recreate) {
    return true;
  }

  // Recreating the pool is only valid once the previous resolve batch has
  // fully drained.
  assert_true(!is_initialized() || !has_pending_resolve_batch());
  Shutdown();

  ID3D12Device* device = provider.GetDevice();

  D3D12_QUERY_HEAP_DESC heap_desc = {};
  heap_desc.Type = D3D12_QUERY_HEAP_TYPE_OCCLUSION;
  heap_desc.Count = requested_capacity;
  heap_desc.NodeMask = 0;

  if (FAILED(device->CreateQueryHeap(&heap_desc, IID_PPV_ARGS(&query_heap_)))) {
    XELOGW(
        "D3D12ZPDQueryPool: Failed to create the ZPD query "
        "heap, falling back to fake sample counts.");
    return false;
  }
  query_heap_->SetName(L"Xenia ZPD QueryHeap");

  // Resolve straight into the readback heap. No extra staging buffer.
  D3D12_RESOURCE_DESC buffer_desc;
  ui::d3d12::util::FillBufferResourceDesc(buffer_desc,
                                          sizeof(uint64_t) * requested_capacity,
                                          D3D12_RESOURCE_FLAG_NONE);

  if (FAILED(device->CreateCommittedResource(
          &ui::d3d12::util::kHeapPropertiesReadback,
          provider.GetHeapFlagCreateNotZeroed(), &buffer_desc,
          D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
          IID_PPV_ARGS(&readback_buffer_)))) {
    XELOGW(
        "D3D12ZPDQueryPool: Failed to allocate the ZPD query "
        "readback buffer, falling back to fake sample counts.");
    Shutdown();
    return false;
  }
  readback_buffer_->SetName(L"Xenia ZPD ReadbackBuffer");

  D3D12_RANGE read_range = {};
  read_range.Begin = 0;
  read_range.End = sizeof(uint64_t) * requested_capacity;

  void* mapping = nullptr;
  if (FAILED(readback_buffer_->Map(0, &read_range, &mapping))) {
    XELOGW(
        "D3D12ZPDQueryPool: Failed to map the ZPD query "
        "readback buffer, falling back to fake sample counts.");
    Shutdown();
    return false;
  }

  readback_mapping_ = reinterpret_cast<uint64_t*>(mapping);
  capacity_ = requested_capacity;

  size_t requested_capacity_rounded = xe::align(requested_capacity, 64u);
  resolve_batch_index_map_.Resize(requested_capacity_rounded);
  // A freshly created pool starts with no queued resolve work.
  resolve_batch_index_map_.Reset();
  resolve_batch_index_count_ = 0;

  free_indices_.clear();
  free_indices_.reserve(requested_capacity);
  for (uint32_t i = requested_capacity; i > 0; --i) {
    free_indices_.push_back(i - 1);
  }
  index_generations_.assign(requested_capacity, 0);

  return true;
}

void D3D12ZPDQueryPool::Shutdown() {
  resolve_batch_index_map_.Resize(0);
  resolve_batch_index_count_ = 0;
  free_indices_.clear();
  index_generations_.clear();
  capacity_ = 0;
  if (readback_mapping_ && readback_buffer_) {
    readback_buffer_->Unmap(0, nullptr);
  }
  readback_mapping_ = nullptr;

  readback_buffer_.Reset();
  query_heap_.Reset();
}

bool D3D12ZPDQueryPool::AcquireQueryIndex(
    uint32_t& query_index, uint32_t& query_generation) {
  if (free_indices_.empty()) {
    query_index = UINT32_MAX;
    query_generation = 0;
    return false;
  }

  query_index = free_indices_.back();
  free_indices_.pop_back();

  assert_true(query_index < index_generations_.size());
  query_generation = ++index_generations_[query_index];
  return true;
}

void D3D12ZPDQueryPool::ReleaseQueryIndex(uint32_t query_index,
                                          uint32_t query_generation) {
  if (query_index >= capacity_) {
    return;
  }

  if (!GenerationMatches(query_index, query_generation)) {
    XELOGW("D3D12ZPDQueryPool: stale release index={} gen={}", query_index,
           query_generation);
    return;
  }

  free_indices_.push_back(query_index);
}

bool D3D12ZPDQueryPool::GenerationMatches(
    uint32_t query_index, uint32_t query_generation) const {
  return query_index < index_generations_.size() &&
         index_generations_[query_index] == query_generation;
}

void D3D12ZPDQueryPool::BeginQuery(DeferredCommandList& deferred_command_list,
                                   uint32_t query_index) const {
  if (!query_heap_ || query_index >= capacity_) {
    return;
  }

  deferred_command_list.D3DBeginQuery(query_heap_.Get(),
                                      D3D12_QUERY_TYPE_OCCLUSION, query_index);
}

void D3D12ZPDQueryPool::EndQuery(DeferredCommandList& deferred_command_list,
                                 uint32_t query_index) const {
  if (!query_heap_ || query_index >= capacity_) {
    return;
  }

  deferred_command_list.D3DEndQuery(query_heap_.Get(),
                                    D3D12_QUERY_TYPE_OCCLUSION, query_index);
}

void D3D12ZPDQueryPool::QueueQueryResolve(uint32_t query_index) {
  if (query_index >= capacity_) {
    return;
  }

  if (resolve_batch_index_map_.AcquireExact(query_index)) {
    ++resolve_batch_index_count_;
  }
}

void D3D12ZPDQueryPool::FlushResolveBatch(
    DeferredCommandList& deferred_command_list, bool submission_open) {
  struct ResolveRange {
    uint32_t start;
    uint32_t count;
  };

  if (!submission_open) {
    return;
  }

  std::vector<ResolveRange> ranges;
  {
    if (!resolve_batch_index_count_) {
      return;
    }

    if (!is_initialized()) {
      resolve_batch_index_map_.Reset();
      resolve_batch_index_count_ = 0;
      return;
    }

    uint32_t range_start = 0;
    uint32_t range_count = 0;
    for (uint32_t index = 0; index < capacity_; ++index) {
      if (!resolve_batch_index_map_.IsAcquired(index)) {
        continue;
      }
      if (range_count == 0) {
        range_start = index;
        range_count = 1;
        continue;
      }
      if (index == range_start + range_count) {
        ++range_count;
        continue;
      }
      ranges.push_back({range_start, range_count});
      range_start = index;
      range_count = 1;
    }
    if (range_count != 0) {
      ranges.push_back({range_start, range_count});
    }

    // Detach the recorded batch now. Later ENDs in the same submission belong
    // to the next pass through this code, not this one.
    resolve_batch_index_map_.Reset();
    resolve_batch_index_count_ = 0;
  }

  if (ranges.empty()) {
    return;
  }

  const uint64_t query_result_stride = kHostZPDResolveStrideBytes;

  for (const ResolveRange& range : ranges) {
    // Contiguous ranges keep command list spam under control.
    deferred_command_list.D3DResolveQueryData(
        query_heap_.Get(), D3D12_QUERY_TYPE_OCCLUSION, range.start, range.count,
        readback_buffer_.Get(), range.start * query_result_stride);
  }
}

uint64_t D3D12ZPDQueryPool::GetQueryReadbackValue(uint32_t query_index) const {
  if (!readback_mapping_ || query_index >= capacity_) {
    return 0;
  }

  uint64_t value = readback_mapping_[query_index];
  return value;
}

}  // namespace d3d12
}  // namespace gpu
}  // namespace xe
