/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_XENOS_OCCLUSION_REPORT_H_
#define XENIA_GPU_XENOS_OCCLUSION_REPORT_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "xenia/base/math.h"
#include "xenia/gpu/xenos.h"

namespace xe {
namespace gpu {

// Small helper for guest occlusion reports and the fixed stuff around them.
// Keep the record math, sentinel checks, and shared report writes here.
// The controller still owns lifetime, reuse, and the pairing mess.
struct XenosOcclusionReport {
  // Guest reports are 0x20-byte records. Titles often keep BEGIN and END
  // close together, but not in a way we can hardcode, so keep the record
  // math simple here and let the controller sort out the pairing.
  static constexpr uint32_t kRecordSizeBytes = 0x20;
  static constexpr uint32_t kRecordAlignMask = ~(kRecordSizeBytes - 1);

  // The common case is two 0x20 records packed into one 0x40 slot.
  // Use this for the cheap checks before falling back to looser pairing.
  static constexpr uint32_t kSlotSizeBytes = 0x40;
  static constexpr uint32_t kSlotAlignMask = ~(kSlotSizeBytes - 1);

  // The page bucket used by the controller's looser pairing checks.
  static constexpr uint32_t kPageSizeBytes = 0x1000;
  static constexpr uint32_t kPageAlignMask = ~(kPageSizeBytes - 1);

  static constexpr uint32_t RecordBase(uint32_t address) {
    return address & kRecordAlignMask;
  }

  // Records in the same slot are close enough that it's worth checking
  // nearby reuse before trying anything fancier.
  static constexpr uint32_t SlotBase(uint32_t address) {
    return address & kSlotAlignMask;
  }

  static constexpr uint32_t PageBase(uint32_t address) {
    return address & kPageAlignMask;
  }

  static constexpr bool IsSameSlot(uint32_t a, uint32_t b) {
    return SlotBase(a) == SlotBase(b);
  }

  static constexpr uint32_t AbsDelta(uint32_t a, uint32_t b) {
    return a > b ? (a - b) : (b - a);
  }

  // Two records sitting in the same 0x40 slot, one half apart from each other.
  static constexpr bool IsCommonHalfSplitPair(uint32_t a, uint32_t b) {
    return IsSameSlot(a, b) && AbsDelta(RecordBase(a), RecordBase(b)) == 0x20;
  }

  // Guest writes this when it wants the report to stay marked pending
  // until we come back with the real value.
  static constexpr uint32_t kPendingSentinelBE = 0xFFFFFEEDu;
  static constexpr uint32_t kPendingSentinelLE = 0xEDFEFFFFu;

  // Sentinel can be written in either endianness, so check for both.
  static constexpr bool IsPendingSentinel(uint32_t value) {
    return value == kPendingSentinelBE || value == kPendingSentinelLE;
  }

  // If any word still contains the guest’s pending sentinel, the report still
  // looks pending.
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

  // Read the BEGIN guest value from a report record.
  // If the guest left the pending sentinel here, just treat it as 0.
  static uint32_t ReadGuestValue(
      const xenos::xe_gpu_depth_sample_counts* report) {
    if (!report) {
      return 0;
    }
    uint32_t value = report->Total_A;
    return IsPendingSentinel(value) ? 0u : value;
  }

  // Keep the B fields at 0 so the guest sees one 32-bit value no matter which
  // field it ends up polling.
  static void WriteGuestValue(xenos::xe_gpu_depth_sample_counts* report,
                              uint32_t value) {
    if (!report) {
      return;
    }
    report->Total_A = value;
    report->Total_B = 0;
    report->ZPass_A = value;
    report->ZPass_B = 0;
    report->ZFail_A = value;
    report->ZFail_B = 0;
    report->StencilFail_A = value;
    report->StencilFail_B = 0;
  }

  // Sometimes we also refresh the BEGIN record so split pairs stop looking
  // pending on both sides and confusing the guest.
  static void WriteGuestBeginEnd(
      xenos::xe_gpu_depth_sample_counts* begin_report,
      xenos::xe_gpu_depth_sample_counts* sink_report, uint32_t delta,
      bool write_begin_report) {
    if (!sink_report) {
      return;
    }
    uint32_t begin_value = ReadGuestValue(begin_report);
    uint32_t end_value = begin_value + delta;
    if (write_begin_report && begin_report && sink_report != begin_report) {
      WriteGuestValue(begin_report, begin_value);
    }
    WriteGuestValue(sink_report, end_value);
  }

  // Host query results in the buffers are 64-bit.
  static constexpr uint32_t kQueryResultStrideBytes = 8;

  // Keep the host pool size cvar sane and round it up to the next power of 2.
  static constexpr int32_t kPoolCapacityMin = 1;
  static constexpr int32_t kPoolCapacityMax = 524288;
  static uint32_t ClampPoolCapacity(int32_t capacity) {
    uint32_t requested_capacity = static_cast<uint32_t>(
        std::clamp(capacity, kPoolCapacityMin, kPoolCapacityMax));
    return xe::next_pow2(requested_capacity);
  }
};

}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_XENOS_OCCLUSION_REPORT_H_
