/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_XENOS_ZPD_REPORT_H_
#define XENIA_GPU_XENOS_ZPD_REPORT_H_

#include <cstddef>
#include <cstdint>

#include "xenia/gpu/xenos.h"

namespace xe {
namespace gpu {

// Helpers for guest ZPD report records.
//
// This keeps record math, sentinel checks, and shared writes in one place.
// Lifetime and pairing stay in the report controller.
struct XenosZPDReport {
  // Reports are 0x20 records.
  static constexpr uint32_t kRecordSizeBytes = 0x20;
  static constexpr uint32_t kRecordAlignMask = ~(kRecordSizeBytes - 1);

  // The common case is two 0x20 records packed into one 0x40 slot.
  static constexpr uint32_t kSlotSizeBytes = 0x40;
  static constexpr uint32_t kSlotAlignMask = ~(kSlotSizeBytes - 1);

  // Page bucket used for grouping.
  static constexpr uint32_t kPageSizeBytes = 0x1000;
  static constexpr uint32_t kPageAlignMask = ~(kPageSizeBytes - 1);

  static constexpr uint32_t GetRecordBase(uint32_t address) {
    return address & kRecordAlignMask;
  }

  // Records in the same slot are close enough for common split checks.
  static constexpr uint32_t GetSlotBase(uint32_t address) {
    return address & kSlotAlignMask;
  }

  static constexpr uint32_t GetPageBase(uint32_t address) {
    return address & kPageAlignMask;
  }

  static constexpr bool IsSameSlot(uint32_t a, uint32_t b) {
    return GetSlotBase(a) == GetSlotBase(b);
  }

  static constexpr uint32_t AbsDelta(uint32_t a, uint32_t b) {
    return a > b ? (a - b) : (b - a);
  }

  // Two records in the same 0x40 slot, one half apart.
  static constexpr bool IsCommonHalfSplitPair(uint32_t a, uint32_t b) {
    return IsSameSlot(a, b) &&
           AbsDelta(GetRecordBase(a), GetRecordBase(b)) == 0x20;
  }

  // Written when the report should stay pending until writeback.
  // Sentinel can show up in either endianness. Perplexingly, some games byte-
  // swap the sentinel in their own code, so both forms are valid.
  static constexpr uint32_t kPendingSentinelBE = 0xFFFFFEEDu;
  static constexpr uint32_t kPendingSentinelLE = 0xEDFEFFFFu;
  static constexpr bool IsPendingSentinel(uint32_t value) {
    return value == kPendingSentinelBE || value == kPendingSentinelLE;
  }

  // The report is pending if any word still contains the pending sentinel.
  static bool IsReportPending(const xenos::xe_gpu_depth_sample_counts* report) {
    if (!report) {
      return false;
    }
    const uint32_t* words = reinterpret_cast<const uint32_t*>(report);
    for (size_t i = 0;
         i < sizeof(xenos::xe_gpu_depth_sample_counts) / sizeof(uint32_t);
         ++i) {
      if (IsPendingSentinel(words[i])) {
        return true;
      }
    }
    return false;
  }

  // Reads the initial sample count. Returns 0 if the report is pending.
  static uint32_t ReadSampleCount(
      const xenos::xe_gpu_depth_sample_counts* report) {
    if (!report) {
      return 0;
    }

    uint32_t sample_count = report->Total_A;
    if (IsPendingSentinel(sample_count)) {
      return 0;
    }
    if (sample_count != 0u) {
      return sample_count;
    }

    sample_count = report->ZPass_A;
    if (IsPendingSentinel(sample_count)) {
      return 0;
    }
    if (sample_count != 0u) {
      return sample_count;
    }

    sample_count = report->ZFail_A;
    return IsPendingSentinel(sample_count) ? 0u : sample_count;
  }

  // Writes one logical 32-bit count. Keeps B at 0.
  static void WriteSampleCount(xenos::xe_gpu_depth_sample_counts* report,
                               uint32_t sample_count) {
    if (!report) {
      return;
    }
    report->Total_A = sample_count;
    report->Total_B = 0;
    report->ZPass_A = sample_count;
    report->ZPass_B = 0;
    report->ZFail_A = sample_count;
    report->ZFail_B = 0;
    report->StencilFail_A = sample_count;
    report->StencilFail_B = 0;
  }

  // Optionally refreshes BEGIN before writing the END delta.
  static void WriteGuestReportDelta(
      xenos::xe_gpu_depth_sample_counts* begin_report,
      xenos::xe_gpu_depth_sample_counts* end_report, uint32_t delta_value,
      bool write_begin_report) {
    if (!end_report) {
      return;
    }
    uint32_t begin_sample_count = ReadSampleCount(begin_report);
    uint32_t end_sample_count = begin_sample_count + delta_value;
    if (write_begin_report && begin_report && end_report != begin_report) {
      WriteSampleCount(begin_report, begin_sample_count);
    }
    WriteSampleCount(end_report, end_sample_count);
  }
};

}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_XENOS_ZPD_REPORT_H_
