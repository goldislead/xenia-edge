/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/gpu/xenos_report_controller.h"

#include <algorithm>
#include <deque>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "xenia/gpu/gpu_flags.h"
#include "xenia/gpu/xenos_occlusion_report.h"

namespace xe {
namespace gpu {
namespace {

struct DeferredWrite {
  XenosReportController::QueryHandle query =
      XenosReportController::kInvalidQuery;
  uint32_t sink_record = 0;
  uint32_t value = 0;
};

bool HasBlockedRecord(const std::vector<uint32_t>& blocked_records,
                      uint32_t record) {
  return std::find(blocked_records.begin(), blocked_records.end(), record) !=
         blocked_records.end();
}

void ErasePairedRecordInMap(std::unordered_map<uint32_t, uint32_t>& map,
                            uint32_t record) {
  std::unordered_map<uint32_t, uint32_t>::iterator it = map.find(record);
  if (it == map.end()) {
    return;
  }

  uint32_t partner = it->second;
  map.erase(it);
  if (!partner) {
    return;
  }

  it = map.find(partner);
  if (it != map.end() && it->second == record) {
    map.erase(it);
  }
}

void SetPairedRecordInMap(std::unordered_map<uint32_t, uint32_t>& map,
                          uint32_t a, uint32_t b) {
  ErasePairedRecordInMap(map, a);
  ErasePairedRecordInMap(map, b);
  map[a] = b;
  map[b] = a;
}

}  // namespace

struct XenosReportController::Impl {
  struct QueryState {
    uint32_t begin_record = 0;
    uint64_t begin_sequence_id = 0;
    bool completed = false;
    uint32_t value = 0;
  };

  struct PendingWrite {
    QueryHandle query = kInvalidQuery;
    uint64_t query_sequence_id = 0;
    uint32_t sink_record = 0;
    uint64_t sink_sequence_id = 0;
    uint32_t mirror_record = 0;
    uint64_t mirror_sequence_id = 0;
  };

  uint64_t GetRecordSequenceLocked(uint32_t record) const {
    std::unordered_map<uint32_t, uint64_t>::const_iterator it =
        record_sequences.find(record);
    return it != record_sequences.end() ? it->second : 0;
  }

  uint32_t GetPairedRecordLocked(uint32_t record) const {
    std::unordered_map<uint32_t, uint32_t>::const_iterator it =
        paired_records.find(record);
    return it != paired_records.end() ? it->second : 0;
  }

  void FlushDeferredWrites(const std::vector<DeferredWrite>& writes) const {
    if (!write_report) {
      return;
    }

    for (const DeferredWrite& write : writes) {
      write_report(write.query, write.sink_record, write.value, context);
    }
  }

  void UpdateLocked(std::vector<DeferredWrite>& writes, Stats* stats) {
    if (pending_writes.empty()) {
      return;
    }

    uint64_t sequence_grace = static_cast<uint64_t>(
        std::max(0, cvars::occlusion_query_fast_sequence_grace));
    std::vector<uint32_t> blocked_records;

    std::deque<PendingWrite>::iterator pending_it = pending_writes.begin();
    while (pending_it != pending_writes.end()) {
      PendingWrite& pending_write = *pending_it;

      std::unordered_map<QueryHandle, QueryState>::iterator query_it =
          queries.find(pending_write.query);
      if (query_it == queries.end()) {
        pending_it = pending_writes.erase(pending_it);
        ++stats->writes_discarded;
        continue;
      }

      QueryState& query_state = query_it->second;
      if (!query_state.completed) {
        blocked_records.push_back(pending_write.sink_record);
        if (pending_write.mirror_record) {
          blocked_records.push_back(pending_write.mirror_record);
        }
        ++pending_it;
        continue;
      }

      if (HasBlockedRecord(blocked_records, pending_write.sink_record) ||
          (pending_write.mirror_record &&
           HasBlockedRecord(blocked_records, pending_write.mirror_record))) {
        ++pending_it;
        continue;
      }

      uint64_t current_begin_sequence =
          GetRecordSequenceLocked(query_state.begin_record);
      uint64_t current_sink_sequence =
          GetRecordSequenceLocked(pending_write.sink_record);

      bool begin_exact_match =
          query_state.begin_sequence_id == current_begin_sequence;
      bool begin_within_grace =
          current_begin_sequence <=
          query_state.begin_sequence_id + sequence_grace;
      bool sink_exact_match =
          pending_write.sink_sequence_id == current_sink_sequence;
      bool sink_within_grace =
          current_sink_sequence <=
          pending_write.sink_sequence_id + sequence_grace;

      if (pending_write.query_sequence_id != query_state.begin_sequence_id ||
          !begin_within_grace || !sink_within_grace) {
        pending_it = pending_writes.erase(pending_it);
        queries.erase(query_it);
        ++stats->writes_discarded_stale;
        continue;
      }

      if ((!begin_exact_match && begin_within_grace) ||
          (!sink_exact_match && sink_within_grace)) {
        ++stats->writes_saved_by_grace;
      }

      DeferredWrite deferred_write;
      deferred_write.query = pending_write.query;
      deferred_write.sink_record = pending_write.sink_record;
      deferred_write.value = query_state.value;
      writes.push_back(deferred_write);

      if (pending_write.mirror_record) {
        uint64_t mirror_sequence =
            GetRecordSequenceLocked(pending_write.mirror_record);
        if (mirror_sequence == pending_write.mirror_sequence_id ||
            mirror_sequence <=
                pending_write.mirror_sequence_id + sequence_grace) {
          deferred_write.sink_record = pending_write.mirror_record;
          writes.push_back(deferred_write);
          ++stats->writes_mirrored;
        } else {
          ++stats->writes_discarded_stale;
        }
      }

      pending_it = pending_writes.erase(pending_it);
      queries.erase(query_it);
      ++stats->writes_retired;
    }
  }

  WriteReport write_report = nullptr;
  void* context = nullptr;

  std::mutex mutex;
  std::deque<PendingWrite> pending_writes;
  std::unordered_map<QueryHandle, QueryState> queries;
  std::unordered_map<uint32_t, uint64_t> record_sequences;
  std::unordered_map<uint32_t, uint32_t> paired_records;
  QueryHandle next_query = 1;
};

XenosReportController::XenosReportController(WriteReport write_report,
                                             void* context)
    : impl_(std::make_unique<Impl>()) {
  impl_->write_report = write_report;
  impl_->context = context;
}

XenosReportController::~XenosReportController() = default;

void XenosReportController::Reset() {
  std::lock_guard<std::mutex> lock(impl_->mutex);

  impl_->pending_writes.clear();
  impl_->queries.clear();
  impl_->record_sequences.clear();
  impl_->paired_records.clear();
  impl_->next_query = 1;
  stats_ = {};
}

XenosReportController::QueryHandle XenosReportController::BeginQuery(
    uint32_t begin_record) {
  std::lock_guard<std::mutex> lock(impl_->mutex);

  if (!begin_record) {
    return kInvalidQuery;
  }

  uint64_t begin_sequence_id = ++impl_->record_sequences[begin_record];
  uint32_t paired_record = impl_->GetPairedRecordLocked(begin_record);
  if (paired_record && paired_record != begin_record) {
    ++impl_->record_sequences[paired_record];
  }

  QueryHandle query = impl_->next_query++;
  if (query == kInvalidQuery) {
    query = impl_->next_query++;
  }

  Impl::QueryState& query_state = impl_->queries[query];
  query_state.begin_record = begin_record;
  query_state.begin_sequence_id = begin_sequence_id;
  query_state.completed = false;
  query_state.value = 0;
  return query;
}

void XenosReportController::ObserveBeginEndPair(uint32_t begin_record,
                                                uint32_t end_record) {
  std::lock_guard<std::mutex> lock(impl_->mutex);

  begin_record = XenosOcclusionReport::RecordBase(begin_record);
  end_record = XenosOcclusionReport::RecordBase(end_record);
  if (!begin_record || !end_record || begin_record == end_record) {
    return;
  }

  if (!XenosOcclusionReport::IsCommonHalfSplitPair(begin_record, end_record)) {
    return;
  }

  SetPairedRecordInMap(impl_->paired_records, begin_record, end_record);
}

void XenosReportController::EnqueueWrite(uint32_t sink_record,
                                         QueryHandle query,
                                         uint32_t mirror_record) {
  std::lock_guard<std::mutex> lock(impl_->mutex);

  if (!sink_record) {
    return;
  }

  std::unordered_map<QueryHandle, Impl::QueryState>::iterator query_it =
      impl_->queries.find(query);
  if (query_it == impl_->queries.end()) {
    ++stats_.writes_discarded;
    return;
  }

  Impl::PendingWrite pending_write;
  pending_write.query = query;
  pending_write.query_sequence_id = query_it->second.begin_sequence_id;
  pending_write.sink_record = sink_record;
  pending_write.sink_sequence_id = impl_->GetRecordSequenceLocked(sink_record);

  if (mirror_record && mirror_record != sink_record) {
    pending_write.mirror_record = mirror_record;
    pending_write.mirror_sequence_id =
        impl_->GetRecordSequenceLocked(mirror_record);
  }

  impl_->pending_writes.push_back(pending_write);
  ++stats_.writes_enqueued;
}

void XenosReportController::MarkQueryCompleted(QueryHandle query,
                                               uint32_t value) {
  std::lock_guard<std::mutex> lock(impl_->mutex);

  std::unordered_map<QueryHandle, Impl::QueryState>::iterator it =
      impl_->queries.find(query);
  if (it == impl_->queries.end()) {
    return;
  }

  it->second.completed = true;
  it->second.value = value;
}

void XenosReportController::Update() {
  std::vector<DeferredWrite> writes;
  {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->UpdateLocked(writes, &stats_);
  }
  impl_->FlushDeferredWrites(writes);
}

void XenosReportController::ResetStats() {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  stats_ = {};
}

}  // namespace gpu
}  // namespace xe
