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
#include <mutex>
#include <unordered_map>
#include <utility>

#include "xenia/gpu/gpu_flags.h"
#include "xenia/gpu/xenos_occlusion_report.h"

namespace xe {
namespace gpu {
namespace {

constexpr uint32_t kMaxLearnedPairDistance = 0x1000;
constexpr uint8_t kSoftPromoteScore = 4;
constexpr uint8_t kSoftDominanceMargin = 2;
constexpr uint8_t kHardPromoteScore = 8;

void SaturatingIncrement(uint8_t& value) {
  if (value != 0xFF) {
    ++value;
  }
}

bool IsPairObservationDominant(uint8_t primary_score, uint8_t secondary_score,
                               uint8_t promote_score,
                               uint8_t dominance_margin) {
  uint16_t secondary =
      static_cast<uint16_t>(secondary_score) + dominance_margin;
  return primary_score >= promote_score && primary_score >= secondary;
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
  };

  struct PairObservation {
    uint32_t primary_partner = 0;
    uint8_t primary_score = 0;
    uint32_t secondary_partner = 0;
    uint8_t secondary_score = 0;
  };

  uint64_t GetRecordSequenceLocked(uint32_t record) const {
    std::unordered_map<uint32_t, uint64_t>::const_iterator it =
        record_sequences.find(record);
    return it != record_sequences.end() ? it->second : 0;
  }

  uint32_t GetPairedRecordLocked(uint32_t record) const {
    std::unordered_map<uint32_t, uint32_t>::const_iterator it =
        paired_records_hard.find(record);
    if (it != paired_records_hard.end()) {
      return it->second;
    }

    it = paired_records_soft.find(record);
    return it != paired_records_soft.end() ? it->second : 0;
  }

  void ObservePairLocked(uint32_t record, uint32_t partner) {
    PairObservation& observation = pair_observations[record];
    if (observation.primary_partner == partner) {
      SaturatingIncrement(observation.primary_score);
    } else if (observation.secondary_partner == partner) {
      SaturatingIncrement(observation.secondary_score);
    } else if (observation.primary_score <= observation.secondary_score) {
      if (!observation.primary_score) {
        observation.primary_partner = partner;
        observation.primary_score = 1;
      } else {
        --observation.primary_score;
      }
    } else {
      if (!observation.secondary_score) {
        observation.secondary_partner = partner;
        observation.secondary_score = 1;
      } else {
        --observation.secondary_score;
      }
    }

    if (observation.secondary_score > observation.primary_score) {
      std::swap(observation.primary_partner, observation.secondary_partner);
      std::swap(observation.primary_score, observation.secondary_score);
    }
  }

  void SetPairedRecordLocked(uint32_t a, uint32_t b, bool hard) {
    if (!a || !b || a == b) {
      return;
    }

    SetPairedRecordInMap(paired_records_soft, a, b);
    if (hard) {
      SetPairedRecordInMap(paired_records_hard, a, b);
    }
  }

  void TryPromotePairLocked(uint32_t record) {
    std::unordered_map<uint32_t, PairObservation>::iterator it =
        pair_observations.find(record);
    if (it == pair_observations.end()) {
      return;
    }

    PairObservation& observation = it->second;
    uint32_t partner = observation.primary_partner;
    if (!partner || partner == record) {
      return;
    }

    if (!IsPairObservationDominant(observation.primary_score,
                                   observation.secondary_score,
                                   kSoftPromoteScore, kSoftDominanceMargin)) {
      return;
    }

    it = pair_observations.find(partner);
    if (it == pair_observations.end()) {
      return;
    }

    PairObservation& partner_observation = it->second;
    if (partner_observation.primary_partner != record ||
        !IsPairObservationDominant(partner_observation.primary_score,
                                   partner_observation.secondary_score,
                                   kSoftPromoteScore, kSoftDominanceMargin)) {
      return;
    }

    std::unordered_map<uint32_t, uint32_t>::iterator hard_it =
        paired_records_hard.find(record);
    if (hard_it != paired_records_hard.end() && hard_it->second != partner) {
      return;
    }

    hard_it = paired_records_hard.find(partner);
    if (hard_it != paired_records_hard.end() && hard_it->second != record) {
      return;
    }

    SetPairedRecordLocked(record, partner, false);

    if (observation.primary_score >= kHardPromoteScore &&
        partner_observation.primary_score >= kHardPromoteScore) {
      SetPairedRecordLocked(record, partner, true);
    }
  }

  void LearnPairLocked(uint32_t begin_record, uint32_t end_record) {
    if (!begin_record || !end_record || begin_record == end_record) {
      return;
    }

    if (XenosOcclusionReport::IsCommonHalfSplitPair(begin_record, end_record)) {
      SetPairedRecordLocked(begin_record, end_record, true);
      return;
    }

    if (XenosOcclusionReport::PageBase(begin_record) !=
        XenosOcclusionReport::PageBase(end_record)) {
      return;
    }
    if (XenosOcclusionReport::AbsDelta(begin_record, end_record) >
        kMaxLearnedPairDistance) {
      return;
    }

    ObservePairLocked(begin_record, end_record);
    ObservePairLocked(end_record, begin_record);
    TryPromotePairLocked(begin_record);
    TryPromotePairLocked(end_record);
  }

  uint32_t ResolveMirrorRecordLocked(uint32_t begin_record,
                                     uint32_t sink_record,
                                     uint32_t mirror_record) const {
    if (mirror_record && mirror_record != sink_record) {
      return mirror_record;
    }

    if (!begin_record || begin_record == sink_record) {
      return 0;
    }

    if (XenosOcclusionReport::IsCommonHalfSplitPair(begin_record,
                                                    sink_record)) {
      return begin_record;
    }

    return GetPairedRecordLocked(sink_record) == begin_record ? begin_record
                                                              : 0;
  }

  std::mutex mutex;
  std::unordered_map<QueryHandle, QueryState> queries;
  std::unordered_map<uint32_t, uint64_t> record_sequences;
  std::unordered_map<uint32_t, uint32_t> paired_records_hard;
  std::unordered_map<uint32_t, uint32_t> paired_records_soft;
  std::unordered_map<uint32_t, PairObservation> pair_observations;
  QueryHandle next_query = 1;
};

XenosReportController::XenosReportController()
    : impl_(std::make_unique<Impl>()) {}

XenosReportController::~XenosReportController() = default;

void XenosReportController::Reset() {
  std::lock_guard<std::mutex> lock(impl_->mutex);

  impl_->queries.clear();
  impl_->record_sequences.clear();
  impl_->paired_records_hard.clear();
  impl_->paired_records_soft.clear();
  impl_->pair_observations.clear();
  impl_->next_query = 1;
  stats_ = {};
}

XenosReportController::QueryHandle XenosReportController::BeginQuery(
    uint32_t begin_address) {
  std::lock_guard<std::mutex> lock(impl_->mutex);

  uint32_t begin_record = XenosOcclusionReport::RecordBase(begin_address);
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
  return query;
}

XenosReportController::PendingWrite XenosReportController::EndQuery(
    QueryHandle query, uint32_t sink_address, bool observe_pair,
    uint32_t mirror_address) {
  std::lock_guard<std::mutex> lock(impl_->mutex);

  PendingWrite pending_write;

  std::unordered_map<QueryHandle, Impl::QueryState>::iterator query_it =
      impl_->queries.find(query);
  if (query_it == impl_->queries.end()) {
    ++stats_.writes_discarded;
    return pending_write;
  }

  uint32_t sink_record = XenosOcclusionReport::RecordBase(sink_address);
  if (!sink_record) {
    ++stats_.writes_discarded;
    return pending_write;
  }

  Impl::QueryState& query_state = query_it->second;
  if (observe_pair) {
    impl_->LearnPairLocked(query_state.begin_record, sink_record);
  }

  uint32_t mirror_record = 0;
  if (mirror_address && mirror_address != sink_address) {
    mirror_record = XenosOcclusionReport::RecordBase(mirror_address);
  }
  mirror_record = impl_->ResolveMirrorRecordLocked(query_state.begin_record,
                                                   sink_record, mirror_record);

  pending_write.query = query;
  pending_write.begin_record = query_state.begin_record;
  pending_write.begin_sequence_id = query_state.begin_sequence_id;
  pending_write.sink_record = sink_record;
  pending_write.sink_sequence_id = impl_->GetRecordSequenceLocked(sink_record);
  if (mirror_record && mirror_record != sink_record) {
    pending_write.mirror_record = mirror_record;
    pending_write.mirror_sequence_id =
        impl_->GetRecordSequenceLocked(mirror_record);
  }
  pending_write.valid = true;

  ++stats_.writes_enqueued;
  return pending_write;
}

XenosReportController::RetiredWrite XenosReportController::CompleteQuery(
    const PendingWrite& pending_write, uint32_t value) {
  std::lock_guard<std::mutex> lock(impl_->mutex);

  RetiredWrite retired_write;
  if (!pending_write.valid || pending_write.query == kInvalidQuery ||
      !pending_write.sink_record) {
    ++stats_.writes_discarded;
    return retired_write;
  }

  std::unordered_map<QueryHandle, Impl::QueryState>::iterator query_it =
      impl_->queries.find(pending_write.query);
  if (query_it == impl_->queries.end()) {
    ++stats_.writes_discarded;
    return retired_write;
  }

  Impl::QueryState query_state = query_it->second;
  impl_->queries.erase(query_it);

  uint64_t sequence_grace = static_cast<uint64_t>(
      std::max(0, cvars::occlusion_query_fast_sequence_grace));

  uint64_t current_begin_sequence =
      impl_->GetRecordSequenceLocked(query_state.begin_record);
  uint64_t current_sink_sequence =
      impl_->GetRecordSequenceLocked(pending_write.sink_record);

  bool begin_exact_match =
      pending_write.begin_sequence_id == current_begin_sequence;
  bool begin_within_grace = current_begin_sequence <=
                            pending_write.begin_sequence_id + sequence_grace;
  bool sink_exact_match =
      pending_write.sink_sequence_id == current_sink_sequence;
  bool sink_within_grace =
      current_sink_sequence <= pending_write.sink_sequence_id + sequence_grace;

  if (pending_write.begin_record != query_state.begin_record ||
      pending_write.begin_sequence_id != query_state.begin_sequence_id ||
      !begin_within_grace || !sink_within_grace) {
    ++stats_.writes_discarded_stale;
    return retired_write;
  }

  if ((!begin_exact_match && begin_within_grace) ||
      (!sink_exact_match && sink_within_grace)) {
    ++stats_.writes_saved_by_grace;
  }

  retired_write.begin_record = query_state.begin_record;
  retired_write.sink_record = pending_write.sink_record;
  retired_write.value = value;
  retired_write.valid = true;

  if (pending_write.mirror_record) {
    uint64_t current_mirror_sequence =
        impl_->GetRecordSequenceLocked(pending_write.mirror_record);
    bool mirror_exact_match =
        pending_write.mirror_sequence_id == current_mirror_sequence;
    bool mirror_within_grace =
    current_mirror_sequence <=
    pending_write.mirror_sequence_id + sequence_grace;
    if (mirror_within_grace) {
      retired_write.mirror_record = pending_write.mirror_record;
      ++stats_.writes_mirrored;
      if (!mirror_exact_match) {
        ++stats_.writes_saved_by_grace;
      }
    } else {
      ++stats_.writes_discarded_stale;
    }
  }

  ++stats_.writes_retired;
  return retired_write;
}

void XenosReportController::AbandonQuery(QueryHandle query) {
  if (query == kInvalidQuery) {
    return;
  }

  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->queries.erase(query);
}

void XenosReportController::ResetStats() {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  stats_ = {};
}

}  // namespace gpu
}  // namespace xe
