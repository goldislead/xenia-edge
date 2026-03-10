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

#include <cstdint>
#include <deque>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace xe {
namespace gpu {

// Coordinates guest report writes with backend resolve completion.
//
// WHY THIS IS NECESSARY:
// Report memory gets reused aggressively. Query slots do too. Once queries go
// async, the backend can't resolve later and write straight back to memory.
// This controller keeps each write tied to the right logical lifetime.
//
// BEGIN starts a logical lifetime and bumps the record sequence. END queues a
// write for that lifetime and sequence. The backend later resolves a delta for
// that lifetime. Queued writes retire in FIFO order when they're safe.
//
// Sequence IDs stop old writes from landing on reused records. If a record base
// is reused before the backend catches up, the old write is dropped.
//
// Some engines split BEGIN and END across different record bases. A common case
// is two 0x20 records sharing one 0x40 slot, one half apart. Those can be
// trusted right away. Soft pairs help with mirrored writes and fast-path hints.
// Hard pairs are used when a new lifetime starts.
//
// TODO(boma): Generalize this to incorporate other report types (screen-extent
// queries) if the approach holds up.
class XenosReportController {
 public:
  using ReportHandle = uint32_t;
  static constexpr ReportHandle kInvalidReportHandle = 0;

  // Called when a queued write still needs more progress.
  using WaitAndResolveCallback = void (*)(ReportHandle report_handle,
                                          void* callback_context);

  // Called when a retired value is ready to write back.
  using CommitGuestReportCallback = void (*)(ReportHandle report_handle,
                                             uint32_t report_record_base,
                                             uint32_t delta_value,
                                             bool write_begin_report,
                                             void* callback_context);

  explicit XenosReportController(
      WaitAndResolveCallback wait_and_resolve_callback,
      CommitGuestReportCallback commit_guest_report_callback,
      void* callback_context)
      : wait_and_resolve_callback_(wait_and_resolve_callback),
        commit_guest_report_callback_(commit_guest_report_callback),
        callback_context_(callback_context) {}

  void Reset();

  // Starts a new lifetime for report_address.
  // Bumps the record sequence. Trusted pairs get bumped too.
  ReportHandle BeginReport(uint32_t report_address);

  // Returns the current sequence ID for the record containing report_address.
  // Kept public for bring-up, logging, and other report paths.
  uint64_t GetRecordSequence(uint32_t report_address) const;

  // Notes that two bases are behaving like a BEGIN / END pair.
  void ObserveBeginEndPair(uint32_t begin_address, uint32_t end_address);

  // Returns the best pairing seen for the record containing report_address.
  // Soft pairs come first. Hard pairs take over if the pattern holds.
  uint32_t GetPairedRecord(uint32_t report_address) const;

  // Queues a record write for later retirement.
  void QueueGuestReportWrite(uint32_t report_address,
                             ReportHandle report_handle,
                             uint32_t mirror_report_address = 0);

  // Called when the delta is ready.
  void SetReportResolved(ReportHandle report_handle, uint32_t delta_value);

  // Pumps queued writes.
  void RetirePendingReports();

  struct Stats {
    uint64_t writes_enqueued = 0;
    uint64_t writes_retired = 0;
    uint64_t writes_discarded = 0;
    uint64_t writes_discarded_stale = 0;
    uint64_t writes_mirrored = 0;
    uint64_t writes_saved_by_grace = 0;
  };

  const Stats& stats() const { return stats_; }
  void ResetStats();

 private:
  // Record write ready to commit outside the lock.
  struct PendingGuestCommit {
    ReportHandle report_handle = kInvalidReportHandle;
    uint32_t report_record_base = 0;
    uint32_t delta_value = 0;
    // Primary END writes refresh BEGIN first. Mirrored writes do not.
    bool write_begin_report = false;
  };

  // Lifetime state tracked separately from query slots.
  struct LogicalReportState {
    uint32_t report_record_base = 0;
    uint64_t record_sequence_id = 0;
    bool resolved = false;
    uint32_t delta_value = 0;
  };

  // Deferred record write waiting on its lifetime.
  struct QueuedReportWrite {
    ReportHandle report_handle = kInvalidReportHandle;
    uint64_t report_record_sequence_id = 0;
    uint32_t report_record_base = 0;
    uint64_t record_sequence_id = 0;
    uint32_t mirror_record_base = 0;
    uint64_t mirror_record_sequence_id = 0;
  };

  // Pairing scores for one record base.
  struct PairObservation {
    uint32_t primary_partner = 0;
    uint8_t primary_score = 0;  // Confidence for the main partner.
    uint32_t secondary_partner = 0;
    uint8_t secondary_score = 0;
  };

  void FlushPendingGuestCommits(
      const std::vector<PendingGuestCommit>& pending_guest_commits);
  void ProcessReportWritesLocked(
      std::vector<PendingGuestCommit>& pending_guest_commits,
      ReportHandle& report_handle_to_resolve);
  void ObservePairLocked(uint32_t report_record_base,
                         uint32_t partner_record_base);
  void SetPairedRecordLocked(uint32_t record_base_a, uint32_t record_base_b,
                             bool hard);
  void TryPromotePairLocked(uint32_t report_record_base);
  uint64_t GetRecordSequenceLocked(uint32_t report_record_base) const;

  WaitAndResolveCallback wait_and_resolve_callback_ = nullptr;
  CommitGuestReportCallback commit_guest_report_callback_ = nullptr;
  void* callback_context_ = nullptr;

  mutable std::mutex mutex_;

  std::deque<QueuedReportWrite> queued_report_writes_;
  std::unordered_map<ReportHandle, LogicalReportState> logical_reports_;

  // Per-record sequence IDs keyed by 0x20 record base.
  std::unordered_map<uint32_t, uint64_t> record_sequences_;

  // Trusted pairings used for sequence bumps.
  std::unordered_map<uint32_t, uint32_t> paired_records_hard_;

  // Learned pairings used for mirrored writes and fast path hints.
  std::unordered_map<uint32_t, uint32_t> paired_records_soft_;

  // Observed pairing candidates and their scores.
  std::unordered_map<uint32_t, PairObservation> record_pair_observations_;
  Stats stats_;
  ReportHandle next_report_handle_ = 1;
};

}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_XENOS_REPORT_CONTROLLER_H_
