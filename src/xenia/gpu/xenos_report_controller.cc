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
#include <unordered_set>

#include "xenia/base/logging.h"
#include "xenia/gpu/gpu_flags.h"
#include "xenia/gpu/xenos_zpd_report.h"

namespace xe {
namespace gpu {

void XenosReportController::Reset() {
  std::lock_guard<std::mutex> lock(mutex_);

  if (cvars::occlusion_query_log) {
    XELOGI("ZPD: Controller reset queued={} logical={}",
           queued_report_writes_.size(), logical_reports_.size());
  }

  queued_report_writes_.clear();
  logical_reports_.clear();
  queued_report_handles_by_slot_.clear();
  slot_sequences_.clear();
  slot_values_.clear();
  stats_ = {};
  next_report_handle_ = 1;
}

XenosReportController::BeginReportResult XenosReportController::BeginReport(
    uint32_t report_address) {
  std::lock_guard<std::mutex> lock(mutex_);

  uint32_t slot_base = XenosZPDReport::GetSlotBase(report_address);
  if (!slot_base) {
    return {};
  }

  uint64_t slot_sequence_id = ++slot_sequences_[slot_base];

  ReportHandle report_handle = next_report_handle_++;
  if (report_handle == kInvalidReportHandle) {
    // 0 is reserved as the invalid handle. Skip over it if the counter wraps.
    report_handle = next_report_handle_++;
  }

  LogicalReportState& report_state = logical_reports_[report_handle];
  report_state.slot_base = slot_base;
  report_state.slot_sequence_id = slot_sequence_id;
  report_state.begin_value = slot_values_[slot_base];
  report_state.resolved = false;
  report_state.delta_value = 0;

  if (cvars::occlusion_query_log) {
    XELOGI(
        "ZPD: Controller BeginReport address=0x{:08X} slot=0x{:08X} "
        "begin_record=0x{:08X} end_record=0x{:08X} seq={} begin_value={}",
        report_address, report_state.slot_base,
        XenosZPDReport::GetBeginRecordBase(slot_base),
        XenosZPDReport::GetEndRecordBase(slot_base),
        report_state.slot_sequence_id, report_state.begin_value);
  }

  return {report_handle, slot_sequence_id, report_state.begin_value};
}

uint64_t XenosReportController::GetSlotSequence(uint32_t report_address) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return GetSlotSequenceLocked(XenosZPDReport::GetSlotBase(report_address));
}

void XenosReportController::QueueGuestReportWrite(
    uint32_t report_address, ReportHandle report_handle) {
  std::lock_guard<std::mutex> lock(mutex_);

  uint32_t slot_base = XenosZPDReport::GetSlotBase(report_address);
  uint32_t end_record_base = XenosZPDReport::GetEndRecordBase(report_address);
  if (!slot_base || !end_record_base) {
    return;
  }

  auto existing_report = logical_reports_.find(report_handle);
  if (existing_report == logical_reports_.end()) {
    ++stats_.writes_discarded;
    return;
  }

  QueuedReportWrite queued_write;
  queued_write.report_handle = report_handle;
  queued_write.slot_base = slot_base;
  queued_write.end_record_base = end_record_base;

  queued_report_writes_.push_back(queued_write);
  queued_report_handles_by_slot_[slot_base].push_back(report_handle);
  ++stats_.writes_enqueued;

  if (cvars::occlusion_query_log) {
    XELOGI(
        "ZPD: Controller QueueGuestReportWrite handle={} slot=0x{:08X} "
        "end_record=0x{:08X} queue_depth={}",
        queued_write.report_handle, queued_write.slot_base,
        queued_write.end_record_base, queued_report_writes_.size());
  }
}

void XenosReportController::SetReportResolved(ReportHandle report_handle,
                                              uint32_t delta_value) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto existing_report = logical_reports_.find(report_handle);
  if (existing_report == logical_reports_.end()) {
    return;
  }

  existing_report->second.resolved = true;
  existing_report->second.delta_value = delta_value;

  if (cvars::occlusion_query_log) {
    XELOGI("ZPD: Controller SetReportResolved handle={} delta={}",
           report_handle, delta_value);
  }
}

uint32_t XenosReportController::RetireResolvedReports() {
  if (cvars::occlusion_query_log) {
    XELOGI("ZPD: Controller RetireResolvedReports begin");
  }

  std::vector<PendingGuestCommit> pending_guest_commits;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    ProcessReportWritesLocked(pending_guest_commits);
  }

  if (cvars::occlusion_query_log) {
    XELOGI("ZPD: Controller RetireResolvedReports commits={}",
           pending_guest_commits.size());
  }

  FlushPendingGuestCommits(pending_guest_commits);
  return static_cast<uint32_t>(pending_guest_commits.size());
}

bool XenosReportController::HasQueuedWriteForAddress(
    uint32_t report_address) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return HasQueuedWriteForSlotLocked(
      XenosZPDReport::GetSlotBase(report_address));
}

void XenosReportController::ResetStats() {
  std::lock_guard<std::mutex> lock(mutex_);
  stats_ = {};
}

void XenosReportController::FlushPendingGuestCommits(
    const std::vector<PendingGuestCommit>& pending_guest_commits) {
  if (!commit_guest_report_callback_) {
    return;
  }

  std::unordered_set<ReportHandle> committed_report_handles;
  for (const PendingGuestCommit& pending_guest_commit : pending_guest_commits) {
    if (!committed_report_handles.insert(pending_guest_commit.report_handle)
             .second) {
      XELOGE("ZPD: Controller FlushPendingGuestCommits duplicate handle={}",
             pending_guest_commit.report_handle);
      continue;
    }
    if (cvars::occlusion_query_log) {
      XELOGI(
          "ZPD: Controller FlushPendingGuestCommits handle={} "
          "record=0x{:08X} begin_value={} delta={} write_begin={}",
          pending_guest_commit.report_handle,
          pending_guest_commit.report_record_base,
          pending_guest_commit.begin_value, pending_guest_commit.delta_value,
          pending_guest_commit.write_begin_report);
    }

    commit_guest_report_callback_(
        pending_guest_commit.report_handle,
        pending_guest_commit.report_record_base,
        pending_guest_commit.begin_value, pending_guest_commit.delta_value,
        pending_guest_commit.write_begin_report, callback_context_);
  }
}

void XenosReportController::ProcessReportWritesLocked(
    std::vector<PendingGuestCommit>& pending_guest_commits) {
  if (queued_report_writes_.empty()) {
    return;
  }

  if (cvars::occlusion_query_log) {
    XELOGI("ZPD: Controller ProcessReportWritesLocked queued={} logical={}",
           queued_report_writes_.size(), logical_reports_.size());
  }

  // Don't let a newer resolved write jump past an older unresolved one on the
  // same slot.
  std::unordered_set<uint32_t> blocked_slot_bases;

  auto write_iterator = queued_report_writes_.begin();
  while (write_iterator != queued_report_writes_.end()) {
    QueuedReportWrite& queued_write = *write_iterator;

    auto existing_report = logical_reports_.find(queued_write.report_handle);
    if (existing_report == logical_reports_.end()) {
      if (cvars::occlusion_query_log) {
        XELOGI("ZPD: Controller ProcessReportWritesLocked drop missing handle={}",
               queued_write.report_handle);
      }
      RemoveQueuedReportHandleForSlotLocked(
          queued_write.slot_base, queued_write.report_handle);
      write_iterator = queued_report_writes_.erase(write_iterator);
      ++stats_.writes_discarded;
      continue;
    }

    LogicalReportState& report_state = existing_report->second;
    if (!report_state.resolved) {
      if (cvars::occlusion_query_log) {
        XELOGI(
            "ZPD: Controller ProcessReportWritesLocked unresolved handle={} "
            "slot=0x{:08X}",
            queued_write.report_handle, queued_write.slot_base);
      }

      blocked_slot_bases.insert(queued_write.slot_base);
      ++write_iterator;
      continue;
    }

    bool blocked =
        blocked_slot_bases.find(queued_write.slot_base) !=
        blocked_slot_bases.end();
    if (blocked) {
      if (cvars::occlusion_query_log) {
        XELOGI(
            "ZPD: Controller ProcessReportWritesLocked blocked handle={} "
            "slot=0x{:08X}",
            queued_write.report_handle, queued_write.slot_base);
      }
      ++write_iterator;
      continue;
    }

    uint64_t current_slot_sequence =
        GetSlotSequenceLocked(report_state.slot_base);
    if (current_slot_sequence != report_state.slot_sequence_id) {
      if (cvars::occlusion_query_log) {
        XELOGI(
            "ZPD: Controller ProcessReportWritesLocked stale handle={} "
            "slot=0x{:08X} report_seq={} current_seq={}",
            queued_write.report_handle, queued_write.slot_base,
            report_state.slot_sequence_id, current_slot_sequence);
      }
      RemoveQueuedReportHandleForSlotLocked(
          queued_write.slot_base, queued_write.report_handle);
      write_iterator = queued_report_writes_.erase(write_iterator);
      logical_reports_.erase(existing_report);
      ++stats_.writes_discarded_stale;
      continue;
    }

    pending_guest_commits.push_back(
        {queued_write.report_handle, queued_write.end_record_base,
         report_state.begin_value, report_state.delta_value,
         report_state.begin_value != 0});

    uint64_t end_value =
        uint64_t(report_state.begin_value) + uint64_t(report_state.delta_value);
    slot_values_[report_state.slot_base] =
        end_value > UINT32_MAX ? UINT32_MAX : uint32_t(end_value);

    if (cvars::occlusion_query_log) {
      XELOGI(
          "ZPD: Controller ProcessReportWritesLocked commit handle={} "
          "slot=0x{:08X} record=0x{:08X} begin_value={} delta={}",
          queued_write.report_handle, queued_write.slot_base,
          queued_write.end_record_base, report_state.begin_value,
          report_state.delta_value);
    }

    RemoveQueuedReportHandleForSlotLocked(
        queued_write.slot_base, queued_write.report_handle);
    write_iterator = queued_report_writes_.erase(write_iterator);
    logical_reports_.erase(existing_report);
    ++stats_.writes_retired;
  }
}

bool XenosReportController::HasQueuedWriteForSlotLocked(
    uint32_t slot_base) const {
  if (!slot_base) {
    return false;
  }

  auto queued_handles_for_slot =
      queued_report_handles_by_slot_.find(slot_base);
  return queued_handles_for_slot != queued_report_handles_by_slot_.end() &&
         !queued_handles_for_slot->second.empty();
}

void XenosReportController::RemoveQueuedReportHandleForSlotLocked(
    uint32_t slot_base, ReportHandle report_handle) {
  auto queued_handles_for_slot =
      queued_report_handles_by_slot_.find(slot_base);
  if (queued_handles_for_slot == queued_report_handles_by_slot_.end()) {
    return;
  }

  std::deque<ReportHandle>& queued_handles = queued_handles_for_slot->second;
  if (!queued_handles.empty() && queued_handles.front() == report_handle) {
    queued_handles.pop_front();
  } else {
    auto queued_handle = std::find(queued_handles.begin(), queued_handles.end(),
                                   report_handle);
    if (queued_handle != queued_handles.end()) {
      queued_handles.erase(queued_handle);
    }
  }

  if (queued_handles.empty()) {
    queued_report_handles_by_slot_.erase(queued_handles_for_slot);
  }
}

uint64_t XenosReportController::GetSlotSequenceLocked(uint32_t slot_base) const {
  auto existing_slot_sequence = slot_sequences_.find(slot_base);
  if (existing_slot_sequence != slot_sequences_.end()) {
    return existing_slot_sequence->second;
  }
  return 0;
}

}  // namespace gpu
}  // namespace xe
