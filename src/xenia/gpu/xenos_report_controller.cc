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
#include <utility>

#include "xenia/base/logging.h"
#include "xenia/gpu/gpu_flags.h"
#include "xenia/gpu/xenos_zpd_report.h"

namespace xe {
namespace gpu {

void XenosReportController::Reset() {
  if (cvars::occlusion_query_log) {
    XELOGE("ZPD: Controller reset queued={} logical={} soft_pairs={} hard_pairs={}",
           queued_report_writes_.size(), logical_reports_.size(),
           paired_records_soft_.size(), paired_records_hard_.size());
  }
  std::lock_guard<std::mutex> lock(mutex_);

  queued_report_writes_.clear();
  logical_reports_.clear();
  record_sequences_.clear();
  paired_records_soft_.clear();
  paired_records_hard_.clear();
  record_pair_observations_.clear();
  stats_ = {};
  next_report_handle_ = 1;
}

XenosReportController::ReportHandle XenosReportController::BeginReport(
    uint32_t report_address) {
  std::lock_guard<std::mutex> lock(mutex_);

  // Track sequences by record base. Raw addresses can move inside the 0x20
  // record.
  uint32_t report_record_base = XenosZPDReport::GetRecordBase(report_address);
  if (!report_record_base) {
    return kInvalidReportHandle;
  }
  uint64_t new_record_sequence = ++record_sequences_[report_record_base];
  if (cvars::occlusion_query_log) {
    XELOGE("ZPD: Controller BeginReport address=0x{:08X} record=0x{:08X} seq={}",
           report_address, report_record_base, new_record_sequence);
  }

  // Only hard pairs bump both sides. Wrong guesses here can stall writes.
  auto existing_hard_pair = paired_records_hard_.find(report_record_base);
  if (existing_hard_pair != paired_records_hard_.end()) {
    uint32_t partner_record_base = existing_hard_pair->second;
    if (partner_record_base && partner_record_base != report_record_base) {
      ++record_sequences_[partner_record_base];
      if (cvars::occlusion_query_log) {
        XELOGE("ZPD: Controller BeginReport hard pair bump record=0x{:08X} partner=0x{:08X} seq={}",
               report_record_base, partner_record_base,
               record_sequences_[partner_record_base]);
      }
    }
  }

  ReportHandle report_handle = next_report_handle_++;
  if (report_handle == kInvalidReportHandle) {
    // 0 is reserved as the invalid handle. Skip over it if the counter wraps.
    report_handle = next_report_handle_++;
  }

  LogicalReportState& report_state = logical_reports_[report_handle];
  report_state.report_record_base = report_record_base;
  report_state.record_sequence_id = new_record_sequence;
  report_state.resolved = false;
  report_state.delta_value = 0;
  if (cvars::occlusion_query_log) {
    XELOGE("ZPD: Controller BeginReport handle={} record=0x{:08X} seq={}",
           report_handle, report_state.report_record_base,
           report_state.record_sequence_id);
  }
  return report_handle;
}

uint64_t XenosReportController::GetRecordSequence(
    uint32_t report_address) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return GetRecordSequenceLocked(XenosZPDReport::GetRecordBase(report_address));
}

void XenosReportController::ObserveBeginEndPair(uint32_t begin_address,
                                                uint32_t end_address) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (cvars::occlusion_query_log) {
    XELOGE("ZPD: Controller ObserveBeginEndPair begin=0x{:08X} end=0x{:08X}",
           begin_address, end_address);
  }

  uint32_t begin_record_base = XenosZPDReport::GetRecordBase(begin_address);
  uint32_t end_record_base = XenosZPDReport::GetRecordBase(end_address);
  if (!begin_record_base || !end_record_base ||
      begin_record_base == end_record_base) {
    if (cvars::occlusion_query_log) {
      XELOGE("ZPD: Controller ObserveBeginEndPair ignored begin_record=0x{:08X} end_record=0x{:08X}",
             begin_record_base, end_record_base);
    }
    return;
  }

  if (XenosZPDReport::IsCommonHalfSplitPair(begin_record_base,
                                            end_record_base)) {
    // Common two-record split in one 0x40 slot. Trust it right away.
    if (cvars::occlusion_query_log) {
      XELOGE("ZPD: Controller ObserveBeginEndPair common split begin_record=0x{:08X} end_record=0x{:08X}",
             begin_record_base, end_record_base);
    }
    SetPairedRecordLocked(begin_record_base, end_record_base, true);
    return;
  }

  // Don't learn pairs across different pages.
  if (XenosZPDReport::GetPageBase(begin_record_base) !=
      XenosZPDReport::GetPageBase(end_record_base)) {
    return;
  }

  constexpr uint32_t kMaxLearnedPairDistance = 0x1000;
  if (XenosZPDReport::AbsDelta(begin_record_base, end_record_base) >
      kMaxLearnedPairDistance) {
    return;
  }

  ObservePairLocked(begin_record_base, end_record_base);
  ObservePairLocked(end_record_base, begin_record_base);
  TryPromotePairLocked(begin_record_base);
  TryPromotePairLocked(end_record_base);
}

uint32_t XenosReportController::GetPairedRecord(uint32_t report_address) const {
  std::lock_guard<std::mutex> lock(mutex_);

  uint32_t report_record_base = XenosZPDReport::GetRecordBase(report_address);
  if (!report_record_base) {
    return 0;
  }

  // Check soft pairs first. Hard pairs are the stricter fallback.
  auto existing_soft_pair = paired_records_soft_.find(report_record_base);
  if (existing_soft_pair != paired_records_soft_.end()) {
    return existing_soft_pair->second;
  }

  auto existing_hard_pair = paired_records_hard_.find(report_record_base);
  if (existing_hard_pair != paired_records_hard_.end()) {
    return existing_hard_pair->second;
  }

  return 0;
}

uint32_t XenosReportController::GetHardPairedRecord(
    uint32_t report_address) const {
  std::lock_guard<std::mutex> lock(mutex_);

  uint32_t report_record_base = XenosZPDReport::GetRecordBase(report_address);
  if (!report_record_base) {
    return 0;
  }

  auto existing_hard_pair = paired_records_hard_.find(report_record_base);
  if (existing_hard_pair != paired_records_hard_.end()) {
    return existing_hard_pair->second;
  }

  return 0;
}

void XenosReportController::QueueGuestReportWrite(
    uint32_t report_address, ReportHandle report_handle,
    uint32_t mirror_report_address) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (cvars::occlusion_query_log) {
    XELOGE("ZPD: Controller QueueGuestReportWrite address=0x{:08X} handle={} mirror=0x{:08X}",
           report_address, report_handle, mirror_report_address);
  }

  uint32_t report_record_base = XenosZPDReport::GetRecordBase(report_address);
  if (!report_record_base) {
    return;
  }

  auto existing_report = logical_reports_.find(report_handle);
  if (existing_report == logical_reports_.end()) {
    // This lifetime is already gone.
    ++stats_.writes_discarded;
    return;
  }

  const LogicalReportState& report_state = existing_report->second;
  QueuedReportWrite queued_write;
  queued_write.report_handle = report_handle;
  queued_write.report_record_sequence_id = report_state.record_sequence_id;
  queued_write.report_record_base = report_record_base;
  queued_write.record_sequence_id = GetRecordSequenceLocked(report_record_base);

  uint32_t mirror_record_base = 0;
  if (mirror_report_address && mirror_report_address != report_address) {
    mirror_record_base = XenosZPDReport::GetRecordBase(mirror_report_address);
  }

  if (mirror_record_base && mirror_record_base != report_record_base) {
    queued_write.mirror_record_base = mirror_record_base;
    queued_write.mirror_record_sequence_id =
        GetRecordSequenceLocked(mirror_record_base);
  }

  // Keep the queue in FIFO order. Safety checks happen later.
  queued_report_writes_.push_back(queued_write);
  ++stats_.writes_enqueued;
  if (cvars::occlusion_query_log) {
    XELOGE("ZPD: Controller QueueGuestReportWrite queued handle={} record=0x{:08X} record_seq={} mirror_record=0x{:08X} mirror_seq={} queue_depth={}",
           queued_write.report_handle, queued_write.report_record_base,
           queued_write.record_sequence_id, queued_write.mirror_record_base,
           queued_write.mirror_record_sequence_id, queued_report_writes_.size());
  }
}

void XenosReportController::SetReportResolved(ReportHandle report_handle,
                                              uint32_t delta_value) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (cvars::occlusion_query_log) {
    XELOGE("ZPD: Controller SetReportResolved handle={} delta={}",
           report_handle, delta_value);
  }

  auto existing_report = logical_reports_.find(report_handle);
  if (existing_report != logical_reports_.end()) {
    existing_report->second.resolved = true;
    existing_report->second.delta_value = delta_value;
  }
}

void XenosReportController::RetirePendingReports() {
  if (cvars::occlusion_query_log) {
    XELOGE("ZPD: Controller RetirePendingReports begin");
  }
  std::vector<PendingGuestCommit> pending_guest_commits;
  ReportHandle report_handle_to_resolve = kInvalidReportHandle;

  {
    std::lock_guard<std::mutex> lock(mutex_);
    ProcessReportWritesLocked(pending_guest_commits, report_handle_to_resolve);
  }

  if (cvars::occlusion_query_log) {
    XELOGE("ZPD: Controller RetirePendingReports commits={} next_wait_handle={}",
           pending_guest_commits.size(), report_handle_to_resolve);
  }
  FlushPendingGuestCommits(pending_guest_commits);
}

void XenosReportController::NudgeReportRetirement(uint32_t report_address) {
  if (!wait_and_resolve_callback_) {
    RetirePendingReports();
    return;
  }

  uint32_t report_record_base = XenosZPDReport::GetRecordBase(report_address);
  if (!report_record_base) {
    return;
  }

  if (cvars::occlusion_query_log) {
    XELOGE("ZPD: Controller NudgeReportRetirement address=0x{:08X} record=0x{:08X}",
           report_address, report_record_base);
  }

  std::vector<PendingGuestCommit> pending_guest_commits;
  ReportHandle report_handle_to_resolve = kInvalidReportHandle;

  {
    std::lock_guard<std::mutex> lock(mutex_);
    ReportHandle ignored_report_handle = kInvalidReportHandle;
    ProcessReportWritesLocked(pending_guest_commits, ignored_report_handle);

    for (const QueuedReportWrite& queued_write : queued_report_writes_) {
      if (!TouchesRecordLocked(queued_write, report_record_base)) {
        continue;
      }
      auto existing_report = logical_reports_.find(queued_write.report_handle);
      if (existing_report == logical_reports_.end()) {
        continue;
      }
      if (!existing_report->second.resolved) {
        report_handle_to_resolve = queued_write.report_handle;
        break;
      }
    }
  }

  if (cvars::occlusion_query_log) {
    XELOGE("ZPD: Controller NudgeReportRetirement commits={} wait_handle={}",
           pending_guest_commits.size(), report_handle_to_resolve);
  }
  FlushPendingGuestCommits(pending_guest_commits);

  if (report_handle_to_resolve == kInvalidReportHandle) {
    return;
  }

  wait_and_resolve_callback_(report_handle_to_resolve, callback_context_);

  pending_guest_commits.clear();
  {
    std::lock_guard<std::mutex> lock(mutex_);
    ReportHandle ignored_report_handle = kInvalidReportHandle;
    ProcessReportWritesLocked(pending_guest_commits, ignored_report_handle);
  }

  if (cvars::occlusion_query_log) {
    XELOGE("ZPD: Controller NudgeReportRetirement after wait commits={}",
           pending_guest_commits.size());
  }
  FlushPendingGuestCommits(pending_guest_commits);
}

bool XenosReportController::TouchesRecordLocked(
    const QueuedReportWrite& queued_write,
    uint32_t report_record_base) const {
  return queued_write.report_record_base == report_record_base ||
         queued_write.mirror_record_base == report_record_base;
}

void XenosReportController::ResetStats() {
  std::lock_guard<std::mutex> lock(mutex_);
  stats_.writes_enqueued = 0;
  stats_.writes_retired = 0;
  stats_.writes_discarded = 0;
  stats_.writes_discarded_stale = 0;
  stats_.writes_mirrored = 0;
  stats_.writes_saved_by_grace = 0;
}

void XenosReportController::FlushPendingGuestCommits(
    const std::vector<PendingGuestCommit>& pending_guest_commits) {
  if (!commit_guest_report_callback_) {
    return;
  }

  // Do the writes outside the lock.
  for (const PendingGuestCommit& pending_guest_commit : pending_guest_commits) {
    if (cvars::occlusion_query_log) {
      XELOGE(
          "ZPD: Controller FlushPendingGuestCommits handle={} "
          "record=0x{:08X} delta={} write_begin={}",
          pending_guest_commit.report_handle,
          pending_guest_commit.report_record_base,
          pending_guest_commit.delta_value,
    }
    commit_guest_report_callback_(pending_guest_commit.report_handle,
                                  pending_guest_commit.report_record_base,
                                  pending_guest_commit.delta_value,
                                  pending_guest_commit.write_begin_report,
                                  callback_context_);
  }
}

void XenosReportController::ProcessReportWritesLocked(
    std::vector<PendingGuestCommit>& pending_guest_commits,
    ReportHandle& report_handle_to_resolve) {
  if (queued_report_writes_.empty()) {
    return;
  }

  if (cvars::occlusion_query_log) {
    XELOGE("ZPD: Controller ProcessReportWritesLocked queued={} logical={}",
           queued_report_writes_.size(), logical_reports_.size());
  }

  report_handle_to_resolve = kInvalidReportHandle;

  uint64_t sequence_grace = static_cast<uint64_t>(
      std::max(0, cvars::occlusion_query_fast_sequence_grace));

  // Don't let a newer resolved write jump past an older unresolved one on
  // the same record.
  std::vector<uint32_t> blocked_record_bases;

  auto write_iterator = queued_report_writes_.begin();
  while (write_iterator != queued_report_writes_.end()) {
    QueuedReportWrite& queued_write = *write_iterator;

    auto existing_report = logical_reports_.find(queued_write.report_handle);
    if (existing_report == logical_reports_.end()) {
      if (cvars::occlusion_query_log) {
        XELOGE("ZPD: Controller ProcessReportWritesLocked drop missing handle={}",
               queued_write.report_handle);
      }
      write_iterator = queued_report_writes_.erase(write_iterator);
      ++stats_.writes_discarded;
      continue;
    }

    LogicalReportState& report_state = existing_report->second;
    if (!report_state.resolved) {
      if (cvars::occlusion_query_log) {
        XELOGE(
            "ZPD: Controller ProcessReportWritesLocked unresolved handle={} "
            "record=0x{:08X} mirror=0x{:08X}",
            queued_write.report_handle, queued_write.report_record_base,
            queued_write.mirror_record_base);
      }
      if (report_handle_to_resolve == kInvalidReportHandle) {
        // Only ask for one unresolved lifetime per pass.
        report_handle_to_resolve = queued_write.report_handle;
      }

      blocked_record_bases.push_back(queued_write.report_record_base);
      // Block mirror writes too. Some engines bounce between two records.
      if (queued_write.mirror_record_base) {
        blocked_record_bases.push_back(queued_write.mirror_record_base);
      }
      ++write_iterator;
      continue;
    }

    auto is_blocked = [&](uint32_t record_base) {
      for (uint32_t blocked_record_base : blocked_record_bases) {
        if (record_base == blocked_record_base) {
          return true;
        }
      }
      return false;
    };

    if (is_blocked(queued_write.report_record_base) ||
        (queued_write.mirror_record_base &&
         is_blocked(queued_write.mirror_record_base))) {
      if (cvars::occlusion_query_log) {
        XELOGE("ZPD: Controller ProcessReportWritesLocked blocked handle={} record=0x{:08X} mirror=0x{:08X}",
               queued_write.report_handle, queued_write.report_record_base,
               queued_write.mirror_record_base);
      }
      ++write_iterator;
      continue;
    }

    uint64_t current_record_sequence =
        GetRecordSequenceLocked(queued_write.report_record_base);

    bool exact_match =
        queued_write.record_sequence_id == current_record_sequence;
    bool within_grace = current_record_sequence <=
                        queued_write.record_sequence_id + sequence_grace;

    // The lifetime still has to match, and the record can't be too far past
    // the grace window.
    if (queued_write.report_record_sequence_id !=
            report_state.record_sequence_id ||
        !within_grace) {
      if (cvars::occlusion_query_log) {
        XELOGE("ZPD: Controller ProcessReportWritesLocked stale handle={} report_seq={} current_report_seq={} record_seq={} current_record_seq={} within_grace={}",
               queued_write.report_handle,
               queued_write.report_record_sequence_id,
               report_state.record_sequence_id, queued_write.record_sequence_id,
               current_record_sequence, within_grace);
      }
      write_iterator = queued_report_writes_.erase(write_iterator);
      logical_reports_.erase(existing_report);
      ++stats_.writes_discarded_stale;
      continue;
    }

    if (!exact_match && within_grace) {
      // The slot moved, but it's still within grace.
      ++stats_.writes_saved_by_grace;
    }

    // Retire mirror writes first. Keep the main END write last.
    if (queued_write.mirror_record_base &&
        queued_write.mirror_record_base != report_state.report_record_base &&
        queued_write.mirror_record_base != queued_write.report_record_base) {
      uint64_t mirror_record_sequence =
          GetRecordSequenceLocked(queued_write.mirror_record_base);
      if (mirror_record_sequence == queued_write.mirror_record_sequence_id ||
          mirror_record_sequence <=
              queued_write.mirror_record_sequence_id + sequence_grace) {
        pending_guest_commits.push_back({queued_write.report_handle,
                                         queued_write.mirror_record_base,
                                         report_state.delta_value, false});
        if (cvars::occlusion_query_log) {
          XELOGE("ZPD: Controller ProcessReportWritesLocked mirror commit handle={} mirror_record=0x{:08X} delta={}",
                 queued_write.report_handle, queued_write.mirror_record_base,
                 report_state.delta_value);
        }
        ++stats_.writes_mirrored;
      } else {
        ++stats_.writes_discarded_stale;
      }
    }

    pending_guest_commits.push_back({queued_write.report_handle,
                                     queued_write.report_record_base,
                                     report_state.delta_value, true});
    if (cvars::occlusion_query_log) {
      XELOGE("ZPD: Controller ProcessReportWritesLocked primary commit handle={} record=0x{:08X} delta={}",
             queued_write.report_handle, queued_write.report_record_base,
             report_state.delta_value);
    }

    write_iterator = queued_report_writes_.erase(write_iterator);
    ++stats_.writes_retired;
  }
}

void XenosReportController::ObservePairLocked(uint32_t report_record_base,
                                              uint32_t partner_record_base) {
  PairObservation& observation = record_pair_observations_[report_record_base];

  // Two entry scoreboard. Good enough for noisy records.
  auto saturation_increment = [](uint8_t& value) {
    if (value != 0xFF) {
      ++value;
    }
  };

  if (observation.primary_partner == partner_record_base) {
    saturation_increment(observation.primary_score);
  } else if (observation.secondary_partner == partner_record_base) {
    saturation_increment(observation.secondary_score);
  } else {
    if (observation.primary_score <= observation.secondary_score) {
      if (observation.primary_score == 0) {
        observation.primary_partner = partner_record_base;
        observation.primary_score = 1;
      } else {
        --observation.primary_score;
      }
    } else {
      if (observation.secondary_score == 0) {
        observation.secondary_partner = partner_record_base;
        observation.secondary_score = 1;
      } else {
        --observation.secondary_score;
      }
    }
  }

  if (cvars::occlusion_query_log) {
    XELOGE("ZPD: Controller ObservePairLocked record=0x{:08X} partner=0x{:08X} primary=0x{:08X}/{} secondary=0x{:08X}/{}",
           report_record_base, partner_record_base, observation.primary_partner,
           observation.primary_score, observation.secondary_partner,
           observation.secondary_score);
  }
  if (observation.secondary_score > observation.primary_score) {
    std::swap(observation.primary_partner, observation.secondary_partner);
    std::swap(observation.primary_score, observation.secondary_score);
  }
}

void XenosReportController::SetPairedRecordLocked(uint32_t record_base_a,
                                                  uint32_t record_base_b,
                                                  bool hard) {
  if (!record_base_a || !record_base_b || record_base_a == record_base_b) {
    return;
  }

  auto set_partner = [&](std::unordered_map<uint32_t, uint32_t>& map) {
    auto erase_partner = [&](uint32_t record_base) {
      auto existing_pair = map.find(record_base);
      if (existing_pair != map.end()) {
        uint32_t old_partner = existing_pair->second;
        map.erase(existing_pair);
        if (old_partner) {
          auto existing_old_partner = map.find(old_partner);
          if (existing_old_partner != map.end() &&
              existing_old_partner->second == record_base) {
            map.erase(existing_old_partner);
          }
        }
      }
    };

    erase_partner(record_base_a);
    erase_partner(record_base_b);
    map[record_base_a] = record_base_b;
    map[record_base_b] = record_base_a;
  };

  set_partner(paired_records_soft_);

  if (hard) {
    set_partner(paired_records_hard_);
  }
  if (cvars::occlusion_query_log) {
    XELOGE("ZPD: Controller SetPairedRecordLocked a=0x{:08X} b=0x{:08X} hard={}",
           record_base_a, record_base_b, hard);
  }
}

void XenosReportController::TryPromotePairLocked(uint32_t report_record_base) {
  auto existing_observation =
      record_pair_observations_.find(report_record_base);
  if (existing_observation == record_pair_observations_.end()) {
    return;
  }

  const PairObservation& observation = existing_observation->second;
  uint32_t partner_record_base = observation.primary_partner;
  if (!partner_record_base || partner_record_base == report_record_base) {
    return;
  }

  // Promote to a soft pair after four steady sightings.
  constexpr uint8_t kSoftPromoteScore = 4;
  constexpr uint8_t kSoftDominanceMargin = 2;

  auto is_dominant = [](const PairObservation& pair_observation) {
    return pair_observation.primary_score >= kSoftPromoteScore &&
           pair_observation.primary_score >=
               static_cast<uint8_t>(pair_observation.secondary_score +
                                    kSoftDominanceMargin);
  };

  if (!is_dominant(observation)) {
    return;
  }

  auto partner_existing_observation =
      record_pair_observations_.find(partner_record_base);
  if (partner_existing_observation == record_pair_observations_.end()) {
    return;
  }
  const PairObservation& partner_observation =
      partner_existing_observation->second;

  if (partner_observation.primary_partner != report_record_base ||
      !is_dominant(partner_observation)) {
    return;
  }

  auto existing_hard_pair = paired_records_hard_.find(report_record_base);
  if (existing_hard_pair != paired_records_hard_.end() &&
      existing_hard_pair->second != partner_record_base) {
    return;
  }
  existing_hard_pair = paired_records_hard_.find(partner_record_base);
  if (existing_hard_pair != paired_records_hard_.end() &&
      existing_hard_pair->second != report_record_base) {
    return;
  }

  if (cvars::occlusion_query_log) {
    XELOGE("ZPD: Controller TryPromotePairLocked soft record=0x{:08X} partner=0x{:08X} score={}/{}",
           report_record_base, partner_record_base, observation.primary_score,
           partner_observation.primary_score);
  }
  SetPairedRecordLocked(report_record_base, partner_record_base, false);

  // Promote to a hard pair after eight steady sightings.
  constexpr uint8_t kHardPromoteScore = 8;
  if (observation.primary_score >= kHardPromoteScore &&
      partner_observation.primary_score >= kHardPromoteScore) {
    if (cvars::occlusion_query_log) {
      XELOGE("ZPD: Controller TryPromotePairLocked hard record=0x{:08X} partner=0x{:08X}",
             report_record_base, partner_record_base);
    }
    SetPairedRecordLocked(report_record_base, partner_record_base, true);
  }
}

uint64_t XenosReportController::GetRecordSequenceLocked(
    uint32_t report_record_base) const {
  auto existing_record_sequence = record_sequences_.find(report_record_base);
  if (existing_record_sequence != record_sequences_.end()) {
    return existing_record_sequence->second;
  }
  return 0;
}

}  // namespace gpu
}  // namespace xe
