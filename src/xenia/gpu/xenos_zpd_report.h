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

#include <algorithm>
#include <cstdint>
#include <cstring>

#include "xenia/gpu/xenos.h"

namespace xe {
namespace gpu {

// One EVENT_WRITE_ZPD report. The hardware has no BEGIN or END, every event
// dumps the free-running sample counters to RB_SAMPLE_COUNT_ADDR, and D3D and
// QueryBatch subtract one report from another in software. Only ZPass can be
// measured on the host, so the failure lanes stay zero.
struct XenosZPDReport {
  uint64_t z_fail = 0;
  uint64_t z_pass = 0;
  uint64_t stencil_fail = 0;
  // Total is the sum of the three outcomes on the hardware too.
  uint64_t total() const { return z_fail + z_pass + stencil_fail; }

  bool operator==(const XenosZPDReport& other) const = default;

  XenosZPDReport& operator+=(const XenosZPDReport& other) {
    z_fail += other.z_fail;
    z_pass += other.z_pass;
    stencil_fail += other.stencil_fail;
    return *this;
  }

  // Native host occlusion query. Only ZPass is measured.
  static XenosZPDReport FromNativeQuery(uint64_t passed) {
    XenosZPDReport report;
    report.z_pass = passed;
    return report;
  }

  // Divides host counts by the draw scale area, rounding to nearest.
  XenosZPDReport Normalized(uint32_t scale_area) const {
    uint64_t scale = scale_area;
    auto normalize = [scale](uint64_t count) {
      return scale <= 1 ? count : (count + (scale >> 1)) / scale;
    };
    XenosZPDReport report;
    report.z_fail = normalize(z_fail);
    report.z_pass = normalize(z_pass);
    report.stencil_fail = normalize(stencil_fail);
    return report;
  }

  // Writes the report to guest memory. GetData wakes on the ZPass lanes and
  // QueryBatch Lock on ZPass_A or StencilFail_B, so those four are written
  // last, in one copy.
  void WriteTo(xenos::xe_gpu_depth_sample_counts* guest) const {
    xenos::xe_gpu_depth_sample_counts values = ToGuest();
    guest->Total_A = values.Total_A;
    guest->Total_B = values.Total_B;
    guest->ZFail_A = values.ZFail_A;
    guest->ZFail_B = values.ZFail_B;
    std::memcpy(
        &guest->ZPass_A, &values.ZPass_A,
        sizeof(values) - offsetof(xenos::xe_gpu_depth_sample_counts, ZPass_A));
  }

  // True if guest memory still holds exactly what WriteTo stored.
  bool Matches(const xenos::xe_gpu_depth_sample_counts* guest) const {
    xenos::xe_gpu_depth_sample_counts values = ToGuest();
    return std::memcmp(guest, &values, sizeof(values)) == 0;
  }

 private:
  // Splits each counter across the A and B lanes. Titles like id Tech 5 mask
  // each lane to 24 bits before summing, so an aggregate-in-A value would
  // cross the mask twice as often as either hardware lane. Low 32 bits only,
  // the hardware counters wrap and so do we.
  xenos::xe_gpu_depth_sample_counts ToGuest() const {
    auto lane_a = [](uint64_t count) {
      return uint32_t(count) - (uint32_t(count) >> 1);
    };
    auto lane_b = [](uint64_t count) { return uint32_t(count) >> 1; };
    xenos::xe_gpu_depth_sample_counts values;
    values.Total_A = lane_a(total());
    values.Total_B = lane_b(total());
    values.ZFail_A = lane_a(z_fail);
    values.ZFail_B = lane_b(z_fail);
    values.ZPass_A = lane_a(z_pass);
    values.ZPass_B = lane_b(z_pass);
    values.StencilFail_A = lane_a(stencil_fail);
    values.StencilFail_B = lane_b(stencil_fail);
    return values;
  }
};

}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_XENOS_ZPD_REPORT_H_
