/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_XENOS_REPORT_CONTROLLER_H_
#define XENIA_GPU_XENOS_REPORT_CONTROLLER_H_

#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace xe {
namespace gpu {

// Coordinates guest report writes with backend resolve completion.
//
// Report memory gets reused aggressively. Once queries go async, the backend
// can't resolve later and write straight back to guest memory. This controller
// keeps each write tied to the right 0x40 ZPD slot lifetime.
//
// BEGIN starts a new slot lifetime and bumps the slot sequence. END queues one
// write for that lifetime. The backend later resolves a delta for the lifetime,
// and queued writes retire in FIFO order when they're safe.
class XenosReportController {
 public:
  using ReportHandle = uint32_t;
  static constexpr ReportHandle kInvalidReportHandle = 0;

  struct BeginReportResult {
    ReportHandle report_handle = kInvalidReportHandle;
    uint64_t slot_sequence_id = 0;
    uint32_t begin_value = 0;
  };

  // Called when a retired value is ready to write back.
  using CommitGuestReportCallback = void (*)(ReportHandle report_handle,
                                             uint32_t report_record_base,
                                             uint32_t begin_value,
                                             uint32_t delta_value,
                                             bool write_begin_report,
                                             void* callback_context);

  explicit XenosReportController(
      CommitGuestReportCallback commit_guest_report_callback,
      void* callback_context)
      : commit_guest_report_callback_(commit_guest_report_callback),
        callback_context_(callback_context) {}

  void Reset();

  // Starts a new lifetime for the slot containing report_address.
  BeginReportResult BeginReport(uint32_t report_address);

  // Returns the current sequence ID for the slot containing report_address.
  uint64_t GetSlotSequence(uint32_t report_address) const;

  // Queues an END-side record write for later retirement.
  void QueueGuestReportWrite(uint32_t report_address,
                             ReportHandle report_handle);

  // Called when the delta is ready.
  void SetReportResolved(ReportHandle report_handle, uint32_t delta_value);

  // Retires any queued writes that are currently resolved.
  // Non-blocking. Returns the number of writes retired.
  uint32_t RetireResolvedReports();

  bool HasQueuedWriteForAddress(uint32_t report_address) const;

  struct Stats {
    uint64_t writes_enqueued = 0;
    uint64_t writes_retired = 0;
    uint64_t writes_discarded = 0;
    uint64_t writes_discarded_stale = 0;
  };

  const Stats& stats() const { return stats_; }
  void ResetStats();

 private:
  // Record write ready to commit outside the lock.
  struct PendingGuestCommit {
    ReportHandle report_handle = kInvalidReportHandle;
    uint32_t report_record_base = 0;
    uint32_t begin_value = 0;
    uint32_t delta_value = 0;
    bool write_begin_report = false;
  };

  // Lifetime state tracked separately from query slots.
  struct LogicalReportState {
    uint32_t slot_base = 0;
    uint64_t slot_sequence_id = 0;
    uint32_t begin_value = 0;
    bool resolved = false;
    uint32_t delta_value = 0;
  };

  // Deferred record write waiting on its lifetime.
  struct QueuedReportWrite {
    ReportHandle report_handle = kInvalidReportHandle;
    uint32_t slot_base = 0;
    uint32_t end_record_base = 0;
  };

  void FlushPendingGuestCommits(
      const std::vector<PendingGuestCommit>& pending_guest_commits);
  void ProcessReportWritesLocked(
      std::vector<PendingGuestCommit>& pending_guest_commits);
  uint64_t GetSlotSequenceLocked(uint32_t slot_base) const;
  bool HasQueuedWriteForSlotLocked(uint32_t slot_base) const;
  void RemoveQueuedReportHandleForSlotLocked(uint32_t slot_base,
                                             ReportHandle report_handle);

  CommitGuestReportCallback commit_guest_report_callback_ = nullptr;
  void* callback_context_ = nullptr;

  mutable std::mutex mutex_;

  std::deque<QueuedReportWrite> queued_report_writes_;
  std::unordered_map<ReportHandle, LogicalReportState> logical_reports_;

  // Per-slot queued report handles in FIFO order.
  std::unordered_map<uint32_t, std::deque<ReportHandle>>
      queued_report_handles_by_slot_;

  // Per-slot sequence IDs keyed by the 0x40 slot base.
  std::unordered_map<uint32_t, uint64_t> slot_sequences_;

  // Last committed cumulative value for each slot.
  std::unordered_map<uint32_t, uint32_t> slot_values_;

  Stats stats_;
  ReportHandle next_report_handle_ = 1;
};

}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_XENOS_REPORT_CONTROLLER_H_
