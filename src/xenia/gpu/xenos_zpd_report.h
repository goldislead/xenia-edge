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
// Lifetime and retirement stay in the report controller.
struct XenosZPDReport {
  // Reports are 0x20 records.
  static constexpr uint32_t kRecordSizeBytes = 0x20;
  static constexpr uint32_t kRecordAlignMask = ~(kRecordSizeBytes - 1);

  // ZPD queries use two 0x20 records in one 0x40 slot.
  static constexpr uint32_t kSlotSizeBytes = 0x40;
  static constexpr uint32_t kSlotAlignMask = ~(kSlotSizeBytes - 1);

  static constexpr uint32_t GetRecordBase(uint32_t address) {
    return address & kRecordAlignMask;
  }

  static constexpr uint32_t GetSlotBase(uint32_t address) {
    return address & kSlotAlignMask;
  }

  // Within a slot, the END record is at the base, and BEGIN is +0x20.
  static constexpr uint32_t GetBeginRecordBase(uint32_t address) {
    uint32_t slot_base = GetSlotBase(address);
    return slot_base ? (slot_base + kRecordSizeBytes) : 0;
  }

  static constexpr uint32_t GetEndRecordBase(uint32_t address) {
    return GetSlotBase(address);
  }

  static constexpr bool IsBeginRecord(uint32_t address) {
    uint32_t record_base = GetRecordBase(address);
    return record_base && record_base == GetBeginRecordBase(record_base);
  }

  static constexpr bool IsEndRecord(uint32_t address) {
    uint32_t record_base = GetRecordBase(address);
    return record_base && record_base == GetEndRecordBase(record_base);
  }

  // Written when the report should stay pending until writeback.
  // Sentinel can show up in either endianness. Some games byte-swap it in
  // their own code, so both forms are valid.
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

  // Reads guest fields in priority order. On real hardware these can be
  // independently written, so we fall through to ZPass_A and ZFail_A if
  // Total_A is 0 but not sentinel.
  static uint32_t ReadSampleCount(
      const xenos::xe_gpu_depth_sample_counts* report) {
    if (!report || IsReportPending(report)) {
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

  // Host APIs only expose a passing sample count, so Total_A mirrors ZPass_A
  // nd the fail/stencil fields are 0.
  static void WriteSampleCount(xenos::xe_gpu_depth_sample_counts* report,
                               uint32_t sample_count) {
    if (!report) {
      return;
    }
    report->Total_A = sample_count;
    report->Total_B = 0;
    report->ZFail_A = 0;
    report->ZFail_B = 0;
    report->ZPass_A = sample_count;
    report->ZPass_B = 0;
    report->StencilFail_A = 0;
    report->StencilFail_B = 0;
  }

  static void WriteGuestReportDeltaWithBeginValue(
      xenos::xe_gpu_depth_sample_counts* begin_report,
      xenos::xe_gpu_depth_sample_counts* end_report, uint32_t begin_value,
      uint32_t delta_value, bool write_begin_report) {
    if (!end_report) {
      return;
    }

    uint64_t end_sample_count = uint64_t(begin_value) + uint64_t(delta_value);
    uint32_t clamped_end_sample_count =
        end_sample_count > UINT32_MAX ? UINT32_MAX : uint32_t(end_sample_count);
    if (write_begin_report && begin_report && end_report != begin_report) {
      WriteSampleCount(begin_report, begin_value);
    }
    WriteSampleCount(end_report, clamped_end_sample_count);
  }

  // Optionally refreshes BEGIN before writing the END delta.
  static void WriteGuestReportDelta(
      xenos::xe_gpu_depth_sample_counts* begin_report,
      xenos::xe_gpu_depth_sample_counts* end_report, uint32_t delta_value,
      bool write_begin_report) {
    uint32_t begin_sample_count = ReadSampleCount(begin_report);
    WriteGuestReportDeltaWithBeginValue(begin_report, end_report,
                                        begin_sample_count, delta_value,
                                        write_begin_report);
  }
};

}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_XENOS_ZPD_REPORT_H_
