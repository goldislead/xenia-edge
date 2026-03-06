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
#include <memory>

namespace xe {
namespace gpu {

// Tracks guest report generations and validates final report writes.
class XenosReportController {
 public:
  using QueryHandle = uint32_t;
  static constexpr QueryHandle kInvalidQuery = 0;

  struct Stats {
    uint64_t writes_enqueued = 0;
    uint64_t writes_retired = 0;
    uint64_t writes_discarded = 0;
    uint64_t writes_discarded_stale = 0;
    uint64_t writes_mirrored = 0;
    uint64_t writes_saved_by_grace = 0;
  };

  struct PendingWrite {
    QueryHandle query = kInvalidQuery;
    uint32_t begin_record = 0;
    uint64_t begin_sequence_id = 0;
    uint32_t sink_record = 0;
    uint64_t sink_sequence_id = 0;
    uint32_t mirror_record = 0;
    uint64_t mirror_sequence_id = 0;
    bool valid = false;
  };

  struct RetiredWrite {
    uint32_t begin_record = 0;
    uint32_t sink_record = 0;
    uint32_t mirror_record = 0;
    uint32_t value = 0;
    bool valid = false;
  };

  XenosReportController();
  ~XenosReportController();

  XenosReportController(const XenosReportController&) = delete;
  XenosReportController& operator=(const XenosReportController&) = delete;

  void Reset();

  QueryHandle BeginQuery(uint32_t begin_address);
  PendingWrite EndQuery(QueryHandle query, uint32_t sink_address,
                        bool observe_pair = true, uint32_t mirror_address = 0);
  RetiredWrite CompleteQuery(const PendingWrite& pending_write, uint32_t value);
  void AbandonQuery(QueryHandle query);

  const Stats& stats() const { return stats_; }
  void ResetStats();

 private:
  struct Impl;

  std::unique_ptr<Impl> impl_;
  Stats stats_;
};

}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_XENOS_REPORT_CONTROLLER_H_
