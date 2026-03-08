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

// Retires guest occlusion report writes in a safe order after the backend has
// finished producing the final values.
class XenosReportController {
 public:
  using QueryHandle = uint32_t;
  static constexpr QueryHandle kInvalidQuery = 0;

  using WriteReport = void (*)(QueryHandle query, uint32_t sink_base,
                               uint32_t value, void* context);

  struct Stats {
    uint64_t writes_enqueued = 0;
    uint64_t writes_retired = 0;
    uint64_t writes_discarded = 0;
    uint64_t writes_discarded_stale = 0;
    uint64_t writes_mirrored = 0;
    uint64_t writes_saved_by_grace = 0;
  };

  XenosReportController(WriteReport write_report, void* context);
  ~XenosReportController();

  XenosReportController(const XenosReportController&) = delete;
  XenosReportController& operator=(const XenosReportController&) = delete;

  void Reset();

  QueryHandle BeginQuery(uint32_t begin_record);
  void ObserveBeginEndPair(uint32_t begin_record, uint32_t end_record);
  void EnqueueWrite(uint32_t sink_record, QueryHandle query,
                    uint32_t mirror_record = 0);
  void MarkQueryCompleted(QueryHandle query, uint32_t value);
  void Update();

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
