/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2022 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/gpu/command_processor.h"

#include "third_party/fmt/include/fmt/format.h"
#include "xenia/base/byte_stream.h"
#include "xenia/base/clock.h"
#include "xenia/base/cvar.h"
#include "xenia/base/logging.h"
#include "xenia/base/profiling.h"
#include "xenia/base/threading.h"
#include "xenia/config.h"
#include "xenia/gpu/gpu_flags.h"
#include "xenia/gpu/graphics_system.h"
#include "xenia/gpu/packet_disassembler.h"
#include "xenia/gpu/sampler_info.h"
#include "xenia/gpu/texture_info.h"
#include "xenia/kernel/kernel_state.h"
#include "xenia/kernel/user_module.h"
#if !defined(NDEBUG)

#define XE_ENABLE_GPU_REG_WRITE_LOGGING 1
#endif
DEFINE_bool(
    log_guest_driven_gpu_register_written_values, false,
    "Only does anything in debug builds, if set will log every write to a gpu "
    "register done by a guest. Does not log writes that are done by the CP on "
    "its own, just ones the guest makes or instructs it to make.",
    "GPU");

DEFINE_bool(disassemble_pm4, false,
            "Only does anything in debug builds, if set will disassemble and "
            "log all PM4 packets sent to the CP.",
            "GPU");

DEFINE_bool(
    log_ringbuffer_kickoff_initiator_bts, false,
    "Only does anything in debug builds, if set will log the pseudo-stacktrace "
    "of the guest thread that wrote the new read position.",
    "GPU");

DEFINE_bool(clear_memory_page_state, true,
            "Refresh state of memory pages to enable gpu written data. "
            "Uses mostly lock-free double-buffering for minimal overhead. "
            "(Disable for minor performance boost, but may break rendering)",
            "GPU");

DEFINE_string(
    readback_resolve, "fast",
    "Controls CPU readback of render-to-texture resolve results.\n"
    " fast: Read from previous frame, copy every frame (default)\n"
    " some: Read from previous frame, skip copy on cache hit\n"
    " full: Wait for GPU to finish (accurate but slow, GPU-CPU sync stall)\n"
    " none: Disable readback completely (some games render better without it)",
    "GPU");

DEFINE_bool(
    readback_memexport, true,
    "Read data written by memory export in shaders on the CPU. "
    "This is needed in some games but many only access exported data on "
    "the GPU, so can be disabled for minor optimization. When "
    "combined with readback_memexport_fast, performance impact is minimal.",
    "GPU");

DEFINE_bool(readback_memexport_fast, true,
            "Use fast (double-buffered, 1 frame delayed) readback for "
            "memexport instead\n"
            "of immediate GPU sync. Removes main performance penalty when "
            "readback_memexport\n"
            "is enabled at the expense of accuracy.",
            "GPU");

namespace xe {
namespace gpu {

// This should be written completely differently with support for different
// types.
void SaveGPUSetting(GPUSetting setting, uint64_t value) {
  switch (setting) {
    case GPUSetting::ClearMemoryPageState:
      OVERRIDE_bool(clear_memory_page_state, static_cast<bool>(value));
      break;
    case GPUSetting::ReadbackMemexport:
      OVERRIDE_bool(readback_memexport, static_cast<bool>(value));
      break;
    case GPUSetting::ReadbackMemexportFast:
      OVERRIDE_bool(readback_memexport_fast, static_cast<bool>(value));
      break;
  }
}

bool GetGPUSetting(GPUSetting setting) {
  switch (setting) {
    case GPUSetting::ClearMemoryPageState:
      return cvars::clear_memory_page_state;
    case GPUSetting::ReadbackMemexport:
      return cvars::readback_memexport;
    case GPUSetting::ReadbackMemexportFast:
      return cvars::readback_memexport_fast;
    default:
      return false;
  }
}

static ReadbackResolveMode ParseReadbackResolveMode() {
  const std::string& mode = cvars::readback_resolve;
  if (mode == "full") {
    return ReadbackResolveMode::kFull;
  } else if (mode == "some") {
    return ReadbackResolveMode::kSome;
  } else if (mode == "none") {
    return ReadbackResolveMode::kDisabled;
  } else {
    // Default to "fast" for any unrecognized value
    return ReadbackResolveMode::kFast;
  }
}

static void SetReadbackResolveCvar(const std::string& mode) {
  OVERRIDE_string(readback_resolve, mode);
}

using namespace xe::gpu::xenos;

CommandProcessor::CommandProcessor(GraphicsSystem* graphics_system,
                                   kernel::KernelState* kernel_state)
    : reader_(nullptr, 0),
      memory_(graphics_system->memory()),
      kernel_state_(kernel_state),
      graphics_system_(graphics_system),
      register_file_(graphics_system_->register_file()),
      trace_writer_(graphics_system->memory()->physical_membase()),
      worker_running_(true),
      write_ptr_index_event_(xe::threading::Event::CreateAutoResetEvent(false)),
      write_ptr_index_(0) {
  assert_not_null(write_ptr_index_event_);
  // Parse and cache readback resolve mode once
  cached_readback_resolve_mode_ = ParseReadbackResolveMode();
}

CommandProcessor::~CommandProcessor() = default;

bool CommandProcessor::Initialize() {
  // Initialize the gamma ramps to their default (linear) values - taken from
  // what games set when starting with the sRGB (return value 1)
  // VdGetCurrentDisplayGamma.
  for (uint32_t i = 0; i < 256; ++i) {
    uint32_t value = i * 0x3FF / 0xFF;
    reg::DC_LUT_30_COLOR& gamma_ramp_entry = gamma_ramp_256_entry_table_[i];
    gamma_ramp_entry.color_10_blue = value;
    gamma_ramp_entry.color_10_green = value;
    gamma_ramp_entry.color_10_red = value;
  }
  for (uint32_t i = 0; i < 128; ++i) {
    reg::DC_LUT_PWL_DATA gamma_ramp_entry = {};
    gamma_ramp_entry.base = (i * 0xFFFF / 0x7F) & ~UINT32_C(0x3F);
    gamma_ramp_entry.delta = i < 0x7F ? 0x200 : 0;
    for (uint32_t j = 0; j < 3; ++j) {
      gamma_ramp_pwl_rgb_[i][j] = gamma_ramp_entry;
    }
  }

  worker_running_ = true;
  worker_thread_ =
      kernel::object_ref<kernel::XHostThread>(new kernel::XHostThread(
          kernel_state_, 128 * 1024, 0,
          [this]() {
            WorkerThreadMain();
            return 0;
          },
          kernel_state_->GetIdleProcess()));
  worker_thread_->set_name("GPU Commands");
  worker_thread_->Create();

  return true;
}

void CommandProcessor::Shutdown() {
  EndTracing();

  worker_running_ = false;
  write_ptr_index_event_->Set();
  worker_thread_->Wait(0, 0, 0, nullptr);
  worker_thread_.reset();
}

void CommandProcessor::InitializeShaderStorage(
    const std::filesystem::path& cache_root, uint32_t title_id, bool blocking,
    std::function<void()> completion_callback) {
  if (completion_callback) {
    completion_callback();
  }
}

void CommandProcessor::RequestFrameTrace(
    const std::filesystem::path& root_path) {
  if (trace_state_ == TraceState::kStreaming) {
    XELOGE("Streaming trace; cannot also trace frame.");
    return;
  }
  if (trace_state_ == TraceState::kSingleFrame) {
    XELOGE("Frame trace already pending; ignoring.");
    return;
  }
  trace_state_ = TraceState::kSingleFrame;
  trace_frame_path_ = root_path;
}

void CommandProcessor::BeginTracing(const std::filesystem::path& root_path) {
  if (trace_state_ == TraceState::kStreaming) {
    XELOGE("Streaming already active; ignoring request.");
    return;
  }
  if (trace_state_ == TraceState::kSingleFrame) {
    XELOGE("Frame trace pending; ignoring streaming request.");
    return;
  }
  // Streaming starts on the next primary buffer execute.
  trace_state_ = TraceState::kStreaming;
  trace_stream_path_ = root_path;
}

void CommandProcessor::EndTracing() {
  if (!trace_writer_.is_open()) {
    return;
  }
  assert_true(trace_state_ == TraceState::kStreaming);
  trace_state_ = TraceState::kDisabled;
  trace_writer_.Close();
}

void CommandProcessor::RestoreRegisters(uint32_t first_register,
                                        const uint32_t* register_values,
                                        uint32_t register_count,
                                        bool execute_callbacks) {
  if (first_register > RegisterFile::kRegisterCount ||
      RegisterFile::kRegisterCount - first_register < register_count) {
    XELOGW(
        "CommandProcessor::RestoreRegisters out of bounds (0x{:X} registers "
        "starting with 0x{:X}, while a total of 0x{:X} registers are stored)",
        register_count, first_register, RegisterFile::kRegisterCount);
    if (first_register > RegisterFile::kRegisterCount) {
      return;
    }
    register_count =
        std::min(uint32_t(RegisterFile::kRegisterCount) - first_register,
                 register_count);
  }
  if (execute_callbacks) {
    for (uint32_t i = 0; i < register_count; ++i) {
      WriteRegister(first_register + i, register_values[i]);
    }
  } else {
    std::memcpy(register_file_->values + first_register, register_values,
                sizeof(uint32_t) * register_count);
  }
}

void CommandProcessor::RestoreGammaRamp(
    const reg::DC_LUT_30_COLOR* new_gamma_ramp_256_entry_table,
    const reg::DC_LUT_PWL_DATA* new_gamma_ramp_pwl_rgb,
    uint32_t new_gamma_ramp_rw_component) {
  std::memcpy(gamma_ramp_256_entry_table_, new_gamma_ramp_256_entry_table,
              sizeof(reg::DC_LUT_30_COLOR) * 256);
  std::memcpy(gamma_ramp_pwl_rgb_, new_gamma_ramp_pwl_rgb,
              sizeof(reg::DC_LUT_PWL_DATA) * 3 * 128);
  gamma_ramp_rw_component_ = new_gamma_ramp_rw_component;
  OnGammaRamp256EntryTableValueWritten();
  OnGammaRampPWLValueWritten();
}

void CommandProcessor::CallInThread(std::function<void()> fn) {
  if (pending_fns_.empty() &&
      kernel::XThread::IsInThread(worker_thread_.get())) {
    fn();
  } else {
    pending_fns_.push(std::move(fn));
  }
}

void CommandProcessor::ClearCaches() {}

void CommandProcessor::InvalidateGpuMemory() {}

void CommandProcessor::ClearReadbackBuffers() {}

void CommandProcessor::SetReadbackResolveMode(ReadbackResolveMode mode) {
  if (cached_readback_resolve_mode_ == mode) {
    return;
  }
  // Update cached value
  cached_readback_resolve_mode_ = mode;
  // Update cvar string for UI display
  const char* mode_str = "fast";
  switch (mode) {
    case ReadbackResolveMode::kDisabled:
      mode_str = "none";
      break;
    case ReadbackResolveMode::kSome:
      mode_str = "some";
      break;
    case ReadbackResolveMode::kFull:
      mode_str = "full";
      break;
    default:
      break;
  }
  SetReadbackResolveCvar(mode_str);

  // Save to per-game config if a title is loaded
  uint32_t title_id = kernel_state_ ? kernel_state_->title_id() : 0;
  if (title_id != 0) {
    toml::table config_table = config::LoadGameConfig(title_id);

    if (!config_table.contains("GPU")) {
      config_table.insert("GPU", toml::table{});
    }

    auto* gpu_table = config_table["GPU"].as_table();
    if (gpu_table) {
      gpu_table->insert_or_assign("readback_resolve", mode_str);
    }

    config::SaveGameConfig(title_id, config_table);
  }
}

void CommandProcessor::SetDesiredSwapPostEffect(
    SwapPostEffect swap_post_effect) {
  if (swap_post_effect_desired_ == swap_post_effect) {
    return;
  }
  swap_post_effect_desired_ = swap_post_effect;
  CallInThread([this, swap_post_effect]() {
    swap_post_effect_actual_ = swap_post_effect;
  });
}

void CommandProcessor::ThrottlePresentation() {
  // Host frame rate limiting based on framerate_limit cvar.
  // This is separate from guest vblank timing (controlled by vsync cvar).
  const uint64_t framerate_limit = cvars::framerate_limit;
  if (framerate_limit == 0) {
    // No host frame limiting
    return;
  }

  const double target_duration_ms =
      1000.0 / static_cast<double>(framerate_limit);
  const uint64_t tick_freq = Clock::guest_tick_frequency();

  const uint64_t target_duration_ticks = static_cast<uint64_t>(
      target_duration_ms * static_cast<double>(tick_freq) / 1000.0);

  // Spin until target duration has elapsed
  while (true) {
    const uint64_t current_time = Clock::QueryGuestTickCount();
    const uint64_t time_delta = current_time - last_swap_time_;

    if (time_delta >= target_duration_ticks) {
      // If we've fallen behind by more than 2 frames, reset to catch up
      if (time_delta > target_duration_ticks * 2) {
        last_swap_time_ = current_time;
      } else {
        last_swap_time_ += target_duration_ticks;
      }
      return;
    }

    const double elapsed_ms = static_cast<double>(time_delta) /
                              (static_cast<double>(tick_freq) / 1000.0);

    const double remaining_ms = target_duration_ms - elapsed_ms;
#if XE_PLATFORM_WIN32
    // Sleep 90% of remaining, spin the rest for accuracy
    const uint64_t sleep_ns =
        static_cast<uint64_t>(remaining_ms * 1000000.0 * 0.90);
    if (sleep_ns > 0) {
      xe::threading::NanoSleep(sleep_ns);
    }
#else
    const uint64_t sleep_ns = static_cast<uint64_t>(remaining_ms * 1000000.0);
    if (sleep_ns > 0) {
      xe::threading::NanoSleep(sleep_ns);
    }
#endif
  }
}

void CommandProcessor::WorkerThreadMain() {
  if (!SetupContext()) {
    xe::FatalError("Unable to setup command processor internal state");
    return;
  }

  while (worker_running_) {
    while (!pending_fns_.empty()) {
      auto fn = std::move(pending_fns_.front());
      pending_fns_.pop();
      fn();
    }

    uint32_t write_ptr_index = write_ptr_index_.load();
    if (write_ptr_index == 0xBAADF00D || read_ptr_index_ == write_ptr_index) {
      SCOPE_profile_cpu_i("gpu", "xe::gpu::CommandProcessor::Stall");
      // We've run out of commands to execute.
      // We spin here waiting for new ones, as the overhead of waiting on our
      // event is too high.
      PrepareForWait();
      uint32_t loop_count = 0;
      do {
        // If we spin around too much, revert to a "low-power" state.
        if (loop_count > 500) {
          constexpr int wait_time_ms = 2;
          xe::threading::Wait(write_ptr_index_event_.get(), true,
                              std::chrono::milliseconds(wait_time_ms));
        } else {
          xe::threading::MaybeYield();
        }
        loop_count++;
        write_ptr_index = write_ptr_index_.load();
      } while (worker_running_ && pending_fns_.empty() &&
               (write_ptr_index == 0xBAADF00D ||
                read_ptr_index_ == write_ptr_index));
      ReturnFromWait();
      if (!worker_running_ || !pending_fns_.empty()) {
        continue;
      }
    }
    assert_true(read_ptr_index_ != write_ptr_index);

    // Execute. Note that we handle wraparound transparently.
    read_ptr_index_ = ExecutePrimaryBuffer(read_ptr_index_, write_ptr_index);

    // TODO(benvanik): use reader->Read_update_freq_ and only issue after moving
    //     that many indices.
    // Keep in mind that the gpu also updates the cpu-side copy if the write
    // pointer and read pointer would be equal
    if (read_ptr_writeback_ptr_) {
      xe::store_and_swap<uint32_t>(
          memory_->TranslatePhysical(read_ptr_writeback_ptr_), read_ptr_index_);
    }

    // FIXME: We're supposed to process the WAIT_UNTIL register at this point,
    // but no games seem to actually use it.
  }

  ShutdownContext();
}

void CommandProcessor::Pause() {
  if (paused_) {
    return;
  }
  paused_ = true;

  threading::Fence fence;
  CallInThread([&fence]() {
    fence.Signal();
    threading::Thread::GetCurrentThread()->Suspend();
  });

  fence.Wait();
}

void CommandProcessor::Resume() {
  if (!paused_) {
    return;
  }
  paused_ = false;

  worker_thread_->thread()->Resume();
}

bool CommandProcessor::Save(ByteStream* stream) {
  assert_true(paused_);

  stream->Write<uint32_t>(primary_buffer_ptr_);
  stream->Write<uint32_t>(primary_buffer_size_);
  stream->Write<uint32_t>(read_ptr_index_);
  stream->Write<uint32_t>(read_ptr_update_freq_);
  stream->Write<uint32_t>(read_ptr_writeback_ptr_);
  stream->Write<uint32_t>(write_ptr_index_.load());

  return true;
}

bool CommandProcessor::Restore(ByteStream* stream) {
  assert_true(paused_);

  primary_buffer_ptr_ = stream->Read<uint32_t>();
  primary_buffer_size_ = stream->Read<uint32_t>();
  read_ptr_index_ = stream->Read<uint32_t>();
  read_ptr_update_freq_ = stream->Read<uint32_t>();
  read_ptr_writeback_ptr_ = stream->Read<uint32_t>();
  write_ptr_index_.store(stream->Read<uint32_t>());

  return true;
}

bool CommandProcessor::SetupContext() {
  active_host_zpd_query_segment_ = {};
  logical_zpd_reports_.clear();
  fast_zpd_report_cached_values_.clear();
  fake_zpd_sample_count_ = 0;
  pending_strict_zpd_retire_handle_ =
      XenosReportController::kInvalidReportHandle;
  pending_strict_zpd_retire_stall_count_ = 0;
  host_zpd_query_resolves_in_flight_.clear();
  zpd_report_controller_ = std::make_unique<XenosReportController>(
      &CommandProcessor::CommitGuestZPDReportCallback, this);
  return true;
}

void CommandProcessor::ShutdownContext() {
  active_host_zpd_query_segment_ = {};
  logical_zpd_reports_.clear();
  fast_zpd_report_cached_values_.clear();
  fake_zpd_sample_count_ = 0;
  pending_strict_zpd_retire_handle_ =
      XenosReportController::kInvalidReportHandle;
  pending_strict_zpd_retire_stall_count_ = 0;
  host_zpd_query_resolves_in_flight_.clear();
  zpd_report_controller_.reset();
}

void CommandProcessor::InitializeRingBuffer(uint32_t ptr, uint32_t size_log2) {
  read_ptr_index_ = 0;
  primary_buffer_ptr_ = ptr;
  primary_buffer_size_ = uint32_t(1) << (size_log2 + 3);

  std::memset(kernel_state_->memory()->TranslatePhysical(primary_buffer_ptr_),
              0, primary_buffer_size_);
}

void CommandProcessor::EnableReadPointerWriteBack(uint32_t ptr,
                                                  uint32_t block_size_log2) {
  // CP_RB_RPTR_ADDR Ring Buffer Read Pointer Address 0x70C
  // ptr = RB_RPTR_ADDR, pointer to write back the address to.
  read_ptr_writeback_ptr_ = ptr;
  // CP_RB_CNTL Ring Buffer Control 0x704
  // block_size = RB_BLKSZ, log2 of number of quadwords read between updates of
  //              the read pointer.
  read_ptr_update_freq_ = uint32_t(1) << block_size_log2 >> 2;
}

XE_NOINLINE XE_COLD void CommandProcessor::LogKickoffInitator(uint32_t value) {
  cpu::backend::GuestPseudoStackTrace st;

  if (logging::ShouldLog(LogLevel::Debug) &&
      kernel_state_->processor()->backend()->PopulatePseudoStacktrace(&st)) {
    logging::LoggerBatch<LogLevel::Debug> log_initiator{};

    log_initiator("Updating read ptr to {}, initiator stacktrace below\n",
                  value);

    for (uint32_t i = 0; i < st.count; ++i) {
      log_initiator("\t{:08X}\n", st.return_addrs[i]);
    }

    if (st.truncated_flag) {
      log_initiator("\t(Truncated stacktrace to {} entries)\n",
                    cpu::backend::MAX_GUEST_PSEUDO_STACKTRACE_ENTRIES);
    }
    log_initiator.submit('d');
  }
}

void CommandProcessor::UpdateWritePointer(uint32_t value) {
  XE_UNLIKELY_IF(cvars::log_ringbuffer_kickoff_initiator_bts) {
    LogKickoffInitator(value);
  }
  write_ptr_index_ = value;
  write_ptr_index_event_->SetBoostPriority();
}

void CommandProcessor::LogRegisterSet(uint32_t register_index, uint32_t value) {
#if XE_ENABLE_GPU_REG_WRITE_LOGGING == 1
  if (cvars::log_guest_driven_gpu_register_written_values &&
      logging::ShouldLog(LogLevel::Debug)) {
    const RegisterInfo* reginfo = RegisterFile::GetRegisterInfo(register_index);

    if (!reginfo) {
      XELOGD("Unknown_Reg{:04X} <- {:08X}\n", register_index, value);
    } else {
      XELOGD("{} <- {:08X}\n", reginfo->name, value);
    }
  }
#endif
}

void CommandProcessor::LogRegisterSets(uint32_t base_register_index,
                                       const uint32_t* values,
                                       uint32_t n_values) {
#if XE_ENABLE_GPU_REG_WRITE_LOGGING == 1
  if (cvars::log_guest_driven_gpu_register_written_values &&
      logging::ShouldLog(LogLevel::Debug)) {
    auto target = logging::internal::GetThreadBuffer();

    auto target_ptr = target.first;

    size_t total_size = 0;

    size_t rem_size = target.second;

    for (uint32_t i = 0; i < n_values; ++i) {
      uint32_t register_index = base_register_index + i;

      uint32_t value = xe::load_and_swap<uint32_t>(&values[i]);

      const RegisterInfo* reginfo =
          RegisterFile::GetRegisterInfo(register_index);

      if (!reginfo) {
        auto tmpres = fmt::format_to_n(target_ptr, rem_size,
                                       "Unknown_Reg{:04X} <- {:08X}\n",
                                       register_index, value);
        target_ptr = tmpres.out;
        rem_size -= tmpres.size;
        total_size += tmpres.size;

      } else {
        auto tmpres = fmt::format_to_n(target_ptr, rem_size, "{} <- {:08X}\n",
                                       reginfo->name, value);
        rem_size -= tmpres.size;
        target_ptr = tmpres.out;
        total_size += tmpres.size;
      }
    }
    logging::internal::AppendLogLine(LogLevel::Debug, 'd', total_size);
  }
#endif
}

void CommandProcessor::HandleSpecialRegisterWrite(uint32_t index,
                                                  uint32_t value) {
  RegisterFile& regs = *register_file_;
  // Scratch register writeback.
  if (index >= XE_GPU_REG_SCRATCH_REG0 && index <= XE_GPU_REG_SCRATCH_REG7) {
    uint32_t scratch_reg = index - XE_GPU_REG_SCRATCH_REG0;
    if ((1 << scratch_reg) & regs.values[XE_GPU_REG_SCRATCH_UMSK]) {
      // Enabled - write to address.
      uint32_t scratch_addr = regs.values[XE_GPU_REG_SCRATCH_ADDR];
      uint32_t mem_addr = scratch_addr + (scratch_reg * 4);
      xe::store_and_swap<uint32_t>(memory_->TranslatePhysical(mem_addr), value);
    }
  } else {
    switch (index) {
      // If this is a COHER register, set the dirty flag.
      // This will block the command processor the next time it WAIT_MEM_REGs
      // and allow us to synchronize the memory.
      case XE_GPU_REG_COHER_STATUS_HOST: {
        regs.values[index] |= UINT32_C(0x80000000);
      } break;

      case XE_GPU_REG_DC_LUT_RW_INDEX: {
        // Reset the sequential read / write component index (see the M56
        // DC_LUT_SEQ_COLOR documentation).
        gamma_ramp_rw_component_ = 0;
      } break;

      case XE_GPU_REG_DC_LUT_SEQ_COLOR: {
        // Should be in the 256-entry table writing mode.
        assert_zero(regs[XE_GPU_REG_DC_LUT_RW_MODE] & 0b1);
        auto gamma_ramp_rw_index = regs.Get<reg::DC_LUT_RW_INDEX>();
        // DC_LUT_SEQ_COLOR is in the red, green, blue order, but the write
        // enable mask is blue, green, red.
        bool write_gamma_ramp_component =
            (regs[XE_GPU_REG_DC_LUT_WRITE_EN_MASK] &
             (UINT32_C(1) << (2 - gamma_ramp_rw_component_))) != 0;
        if (write_gamma_ramp_component) {
          reg::DC_LUT_30_COLOR& gamma_ramp_entry =
              gamma_ramp_256_entry_table_[gamma_ramp_rw_index.rw_index];
          // Bits 0:5 are hardwired to zero.
          uint32_t gamma_ramp_seq_color =
              regs.Get<reg::DC_LUT_SEQ_COLOR>().seq_color >> 6;
          switch (gamma_ramp_rw_component_) {
            case 0:
              gamma_ramp_entry.color_10_red = gamma_ramp_seq_color;
              break;
            case 1:
              gamma_ramp_entry.color_10_green = gamma_ramp_seq_color;
              break;
            case 2:
              gamma_ramp_entry.color_10_blue = gamma_ramp_seq_color;
              break;
          }
        }
        if (++gamma_ramp_rw_component_ >= 3) {
          gamma_ramp_rw_component_ = 0;
          reg::DC_LUT_RW_INDEX new_gamma_ramp_rw_index = gamma_ramp_rw_index;
          ++new_gamma_ramp_rw_index.rw_index;
          WriteRegister(
              XE_GPU_REG_DC_LUT_RW_INDEX,
              xe::memory::Reinterpret<uint32_t>(new_gamma_ramp_rw_index));
        }
        if (write_gamma_ramp_component) {
          OnGammaRamp256EntryTableValueWritten();
        }
      } break;

      case XE_GPU_REG_DC_LUT_PWL_DATA: {
        // Should be in the PWL writing mode.
        assert_not_zero(regs[XE_GPU_REG_DC_LUT_RW_MODE] & 0b1);
        auto gamma_ramp_rw_index = regs.Get<reg::DC_LUT_RW_INDEX>();
        // Bit 7 of the index is ignored for PWL.
        uint32_t gamma_ramp_rw_index_pwl = gamma_ramp_rw_index.rw_index & 0x7F;
        // DC_LUT_PWL_DATA is likely in the red, green, blue order because
        // DC_LUT_SEQ_COLOR is, but the write enable mask is blue, green, red.
        bool write_gamma_ramp_component =
            (regs[XE_GPU_REG_DC_LUT_WRITE_EN_MASK] &
             (UINT32_C(1) << (2 - gamma_ramp_rw_component_))) != 0;
        if (write_gamma_ramp_component) {
          reg::DC_LUT_PWL_DATA& gamma_ramp_entry =
              gamma_ramp_pwl_rgb_[gamma_ramp_rw_index_pwl]
                                 [gamma_ramp_rw_component_];
          auto gamma_ramp_value = regs.Get<reg::DC_LUT_PWL_DATA>();
          // Bits 0:5 are hardwired to zero.
          gamma_ramp_entry.base = gamma_ramp_value.base & ~UINT32_C(0x3F);
          gamma_ramp_entry.delta = gamma_ramp_value.delta & ~UINT32_C(0x3F);
        }
        if (++gamma_ramp_rw_component_ >= 3) {
          gamma_ramp_rw_component_ = 0;
          reg::DC_LUT_RW_INDEX new_gamma_ramp_rw_index = gamma_ramp_rw_index;
          // TODO(Triang3l): Should this increase beyond 7 bits for PWL?
          // Direct3D 9 explicitly sets rw_index to 0x80 after writing the last
          // PWL entry. However, the DC_LUT_RW_INDEX documentation says that for
          // PWL, the bit 7 is ignored.
          new_gamma_ramp_rw_index.rw_index =
              (gamma_ramp_rw_index.rw_index & ~UINT32_C(0x7F)) |
              ((gamma_ramp_rw_index_pwl + 1) & 0x7F);
          WriteRegister(
              XE_GPU_REG_DC_LUT_RW_INDEX,
              xe::memory::Reinterpret<uint32_t>(new_gamma_ramp_rw_index));
        }
        if (write_gamma_ramp_component) {
          OnGammaRampPWLValueWritten();
        }
      } break;

      case XE_GPU_REG_DC_LUT_30_COLOR: {
        // Should be in the 256-entry table writing mode.
        assert_zero(regs[XE_GPU_REG_DC_LUT_RW_MODE] & 0b1);
        auto gamma_ramp_rw_index = regs.Get<reg::DC_LUT_RW_INDEX>();
        uint32_t gamma_ramp_write_enable_mask =
            regs[XE_GPU_REG_DC_LUT_WRITE_EN_MASK] & 0b111;
        if (gamma_ramp_write_enable_mask) {
          reg::DC_LUT_30_COLOR& gamma_ramp_entry =
              gamma_ramp_256_entry_table_[gamma_ramp_rw_index.rw_index];
          auto gamma_ramp_value = regs.Get<reg::DC_LUT_30_COLOR>();
          if (gamma_ramp_write_enable_mask & 0b001) {
            gamma_ramp_entry.color_10_blue = gamma_ramp_value.color_10_blue;
          }
          if (gamma_ramp_write_enable_mask & 0b010) {
            gamma_ramp_entry.color_10_green = gamma_ramp_value.color_10_green;
          }
          if (gamma_ramp_write_enable_mask & 0b100) {
            gamma_ramp_entry.color_10_red = gamma_ramp_value.color_10_red;
          }
        }
        // TODO(Triang3l): Should this reset the component write index? If this
        // increase is assumed to behave like a full DC_LUT_RW_INDEX write, it
        // probably should. Currently this also calls WriteRegister for
        // DC_LUT_RW_INDEX, which resets gamma_ramp_rw_component_ as well.
        gamma_ramp_rw_component_ = 0;
        reg::DC_LUT_RW_INDEX new_gamma_ramp_rw_index = gamma_ramp_rw_index;
        ++new_gamma_ramp_rw_index.rw_index;
        WriteRegister(
            XE_GPU_REG_DC_LUT_RW_INDEX,
            xe::memory::Reinterpret<uint32_t>(new_gamma_ramp_rw_index));
        if (gamma_ramp_write_enable_mask) {
          OnGammaRamp256EntryTableValueWritten();
        }
      } break;
    }
  }
}
void CommandProcessor::WriteRegister(uint32_t index, uint32_t value) {
  // chrispy: rearrange check order, place set after checks

  if (XE_LIKELY(index < RegisterFile::kRegisterCount)) {
    register_file_->values[index] = value;

    // quick pre-test
    // todo: figure out just how unlikely this is. if very (it ought to be,
    // theres a ton of registers other than these) make this predicate
    // branchless and mark with unlikely, then make HandleSpecialRegisterWrite
    // noinline yep, its very unlikely. these ORS here are meant to be bitwise
    // ors, so that we do not do branching evaluation of the conditions (we will
    // almost always take all of the branches)

    unsigned expr = (index - XE_GPU_REG_SCRATCH_REG0 < 8) |
                    (index == XE_GPU_REG_COHER_STATUS_HOST) |
                    ((index - XE_GPU_REG_DC_LUT_RW_INDEX) <=
                     (XE_GPU_REG_DC_LUT_30_COLOR - XE_GPU_REG_DC_LUT_RW_INDEX));
    // chrispy: reordered for msvc branch probability (assumes if is taken and
    // else is not)
    if (XE_LIKELY(expr == 0)) {
      XE_MSVC_REORDER_BARRIER();

    } else {
      HandleSpecialRegisterWrite(index, value);
    }
  } else {
    XELOGW("CommandProcessor::WriteRegister index out of bounds: {}", index);
    return;
  }
}
void CommandProcessor::WriteRegistersFromMem(uint32_t start_index,
                                             uint32_t* base,
                                             uint32_t num_registers) {
  for (uint32_t i = 0; i < num_registers; ++i) {
    uint32_t data = xe::load_and_swap<uint32_t>(base + i);
    this->WriteRegister(start_index + i, data);
  }
}

void CommandProcessor::WriteRegisterRangeFromRing(xe::RingBuffer* ring,
                                                  uint32_t base,
                                                  uint32_t num_registers) {
  for (uint32_t i = 0; i < num_registers; ++i) {
    uint32_t data = ring->ReadAndSwap<uint32_t>();
    WriteRegister(base + i, data);
  }
}

void CommandProcessor::WriteALURangeFromRing(xe::RingBuffer* ring,
                                             uint32_t base,
                                             uint32_t num_times) {
  WriteRegisterRangeFromRing(ring, base + 0x4000, num_times);
}

void CommandProcessor::WriteFetchRangeFromRing(xe::RingBuffer* ring,
                                               uint32_t base,
                                               uint32_t num_times) {
  WriteRegisterRangeFromRing(ring, base + 0x4800, num_times);
}

void CommandProcessor::WriteBoolRangeFromRing(xe::RingBuffer* ring,
                                              uint32_t base,
                                              uint32_t num_times) {
  WriteRegisterRangeFromRing(ring, base + 0x4900, num_times);
}

void CommandProcessor::WriteLoopRangeFromRing(xe::RingBuffer* ring,
                                              uint32_t base,
                                              uint32_t num_times) {
  WriteRegisterRangeFromRing(ring, base + 0x4908, num_times);
}

void CommandProcessor::WriteREGISTERSRangeFromRing(xe::RingBuffer* ring,
                                                   uint32_t base,
                                                   uint32_t num_times) {
  WriteRegisterRangeFromRing(ring, base + 0x2000, num_times);
}

void CommandProcessor::WriteALURangeFromMem(uint32_t start_index,
                                            uint32_t* base,
                                            uint32_t num_registers) {
  WriteRegistersFromMem(start_index + 0x4000, base, num_registers);
}

void CommandProcessor::WriteFetchRangeFromMem(uint32_t start_index,
                                              uint32_t* base,
                                              uint32_t num_registers) {
  WriteRegistersFromMem(start_index + 0x4800, base, num_registers);
}

void CommandProcessor::WriteBoolRangeFromMem(uint32_t start_index,
                                             uint32_t* base,
                                             uint32_t num_registers) {
  WriteRegistersFromMem(start_index + 0x4900, base, num_registers);
}

void CommandProcessor::WriteLoopRangeFromMem(uint32_t start_index,
                                             uint32_t* base,
                                             uint32_t num_registers) {
  WriteRegistersFromMem(start_index + 0x4908, base, num_registers);
}

void CommandProcessor::WriteREGISTERSRangeFromMem(uint32_t start_index,
                                                  uint32_t* base,
                                                  uint32_t num_registers) {
  WriteRegistersFromMem(start_index + 0x2000, base, num_registers);
}
XE_NOINLINE
void CommandProcessor::WriteOneRegisterFromRing(uint32_t base,
                                                uint32_t num_times) {
  for (uint32_t m = 0; m < num_times; m++) {
    uint32_t reg_data = reader_.ReadAndSwap<uint32_t>();
    uint32_t target_index = base;
    WriteRegister(target_index, reg_data);
  }
}
void CommandProcessor::MakeCoherent() {
  SCOPE_profile_cpu_f("gpu");

  // Status host often has 0x01000000 or 0x03000000.
  // This is likely toggling VC (vertex cache) or TC (texture cache).
  // Or, it also has a direction in here maybe - there is probably
  // some way to check for dest coherency (what all the COHER_DEST_BASE_*
  // registers are for).
  // Best docs I've found on this are here:
  // https://web.archive.org/web/20160711162346/https://amd-dev.wpengine.netdna-cdn.com/wordpress/media/2013/10/R6xx_R7xx_3D.pdf
  // https://cgit.freedesktop.org/xorg/driver/xf86-video-radeonhd/tree/src/r6xx_accel.c?id=3f8b6eccd9dba116cc4801e7f80ce21a879c67d2#n454

  volatile uint32_t* regs_volatile = register_file_->values;
  auto status_host = xe::memory::Reinterpret<reg::COHER_STATUS_HOST>(
      uint32_t(regs_volatile[XE_GPU_REG_COHER_STATUS_HOST]));
  uint32_t base_host = regs_volatile[XE_GPU_REG_COHER_BASE_HOST];
  uint32_t size_host = regs_volatile[XE_GPU_REG_COHER_SIZE_HOST];

  if (!status_host.status) {
    return;
  }

  const char* action = "N/A";
  if (status_host.vc_action_ena && status_host.tc_action_ena) {
    action = "VC | TC";
  } else if (status_host.tc_action_ena) {
    action = "TC";
  } else if (status_host.vc_action_ena) {
    action = "VC";
  }

  // TODO(benvanik): notify resource cache of base->size and type.
  XELOGGPU("Make {:08X} -> {:08X} ({}b) coherent, action = {}", base_host,
           base_host + size_host, size_host, action);

  // Mark coherent.
  regs_volatile[XE_GPU_REG_COHER_STATUS_HOST] = 0;
}

void CommandProcessor::PrepareForWait() { trace_writer_.Flush(); }

void CommandProcessor::ReturnFromWait() {}

void CommandProcessor::InitializeTrace() {
  // Write the initial register values, to be loaded directly into the
  // RegisterFile since all registers, including those that may have side
  // effects on setting, will be saved.
  trace_writer_.WriteRegisters(
      0, reinterpret_cast<const uint32_t*>(register_file_->values),
      RegisterFile::kRegisterCount, false);

  trace_writer_.WriteGammaRamp(gamma_ramp_256_entry_table(),
                               gamma_ramp_pwl_rgb(), gamma_ramp_rw_component_);
}

bool CommandProcessor::BeginGuestZPDReport(uint32_t report_address) {
  if (!cvars::occlusion_query_enable || !zpd_report_controller_) {
    return false;
  }

  uint32_t carried_cached_delta = 0;
  uint32_t carried_from_slot_base = 0;

  EnsureZPDHostQueryResources();
  if (!IsHostZPDQueryPoolReady()) {
    return false;
  }

  if (active_host_zpd_query_segment_.logical_active) {
    if (active_host_zpd_query_segment_.end_record) {
      EndGuestZPDReport(active_host_zpd_query_segment_.end_record, true);
    } else {
      if (cvars::occlusion_query_log) {
        XELOGI(
            "ZPD: BeginGuestZPDReport forcing close without end record "
            "handle={}",
            active_host_zpd_query_segment_.report_handle);
      }
      carried_from_slot_base = active_host_zpd_query_segment_.slot_base;
      guest_zpd_report_stats_.forced_close_no_end_record++;
      auto dying_logical_report = logical_zpd_reports_.find(
          active_host_zpd_query_segment_.report_handle);
      // If a previous logical report is being forced closed without an END,
      // carry its last cached delta forward so the new lifetime doesn't start
      // from 0. Guests that poll the same slot in a tight loop would otherwise
      // see a bad zero result.
      if (dying_logical_report != logical_zpd_reports_.end()) {
        carried_cached_delta = dying_logical_report->second.cached_delta;
      }
      if (active_host_zpd_query_segment_.segment_active) {
        if (DiscardHostZPDQuery(
                active_host_zpd_query_segment_.query_index,
                active_host_zpd_query_segment_.query_generation)) {
          guest_zpd_report_stats_.segments_ended++;
        } else {
          guest_zpd_report_stats_.failed++;
        }
      }
      logical_zpd_reports_.erase(active_host_zpd_query_segment_.report_handle);
      active_host_zpd_query_segment_ = {};
    }
  }

  uint32_t slot_base = XenosZPDReport::GetSlotBase(report_address);
  uint32_t begin_record = XenosZPDReport::GetBeginRecordBase(slot_base);
  uint32_t end_record = XenosZPDReport::GetEndRecordBase(slot_base);
  XenosReportController::BeginReportResult begin_report_result =
      zpd_report_controller_->BeginReport(report_address);
  XenosReportController::ReportHandle report_handle =
      begin_report_result.report_handle;
  if (report_handle == XenosReportController::kInvalidReportHandle) {
    if (cvars::occlusion_query_log) {
      XELOGI("ZPD: BeginGuestZPDReport controller rejected address=0x{:08X}",
             report_address);
    }
    return false;
  }

  uint64_t slot_sequence_id = begin_report_result.slot_sequence_id;

  LogicalGuestZPDReport& logical = logical_zpd_reports_[report_handle];
  logical.slot_sequence_id = slot_sequence_id;
  logical.slot_base = slot_base;
  logical.begin_record = begin_record;
  logical.end_record = end_record;
  logical.begin_value = begin_report_result.begin_value;
  logical.accumulated_samples = 0;
  logical.last_segment_end_submission = 0;
  logical.pending_segments = 0;
  logical.cached_delta = 0;
  logical.ended = false;

  if (slot_base == carried_from_slot_base && carried_cached_delta != 0) {
    logical.cached_delta = carried_cached_delta;
    // Re-stamp the orphan END cache with the new sequence so a cached hit
    // survives the controller's slot sequence bump for the new lifetime.
    fast_zpd_report_cached_values_[end_record] = {carried_cached_delta,
                                                  slot_sequence_id};
    guest_zpd_report_stats_.cached_delta_carried++;
  }

  active_host_zpd_query_segment_.report_handle = report_handle;
  active_host_zpd_query_segment_.slot_base = slot_base;
  active_host_zpd_query_segment_.begin_record = begin_record;
  active_host_zpd_query_segment_.end_record = end_record;
  active_host_zpd_query_segment_.query_index = UINT32_MAX;
  active_host_zpd_query_segment_.query_generation = 0;
  active_host_zpd_query_segment_.segment_active = false;
  active_host_zpd_query_segment_.segment_pending_begin = true;
  active_host_zpd_query_segment_.logical_active = true;

  if (cvars::occlusion_query_log) {
    XELOGI(
        "ZPD: BeginGuestZPDReport address=0x{:08X} slot=0x{:08X} "
        "begin_record=0x{:08X} end_record=0x{:08X} handle={}",
        report_address, slot_base, begin_record, end_record, report_handle);
  }

  guest_zpd_report_stats_.logical_begun++;
  ResumeActiveHostZPDQuerySegment(true);
  return true;
}

bool CommandProcessor::EndGuestZPDReport(uint32_t report_address,
                                         bool guest_forced_end) {
  if (!cvars::occlusion_query_enable || !zpd_report_controller_ ||
      !active_host_zpd_query_segment_.logical_active) {
    return false;
  }

  XenosReportController::ReportHandle report_handle =
      active_host_zpd_query_segment_.report_handle;
  uint32_t stored_end_record = active_host_zpd_query_segment_.end_record;
  uint32_t report_record_base = XenosZPDReport::GetRecordBase(report_address);
  if (!report_record_base) {
    report_record_base = stored_end_record;
  }

  if (active_host_zpd_query_segment_.segment_active) {
    SplitActiveHostZPDQuerySegment();
  }
  active_host_zpd_query_segment_.segment_pending_begin = false;

  if (!report_record_base) {
    logical_zpd_reports_.erase(report_handle);
    if (cvars::occlusion_query_log) {
      XELOGI(
          "ZPD: EndGuestZPDReport dropping handle={} with unknown record "
          "base forced={}",
          report_handle, guest_forced_end);
    }
    active_host_zpd_query_segment_ = {};
    return false;
  }

  bool resolved_immediately = false;
  bool missing_logical_report = false;
  uint32_t begin_record = 0;
  uint32_t begin_value = 0;
  uint32_t pending_segments = 0;
  uint32_t final_value = 0;
  uint32_t cached_delta =
      static_cast<uint32_t>(cvars::occlusion_query_fast_cached_delta);

  auto it = logical_zpd_reports_.find(report_handle);
  if (it == logical_zpd_reports_.end()) {
    missing_logical_report = true;
  } else {
    LogicalGuestZPDReport& logical = it->second;
    logical.ended = true;
    logical.end_record = report_record_base;
    begin_record = logical.begin_record;
    begin_value = logical.begin_value;
    pending_segments = logical.pending_segments;

    if (logical.pending_segments == 0) {
      resolved_immediately = true;
      final_value = NormalizeZPDReportSampleCount(logical.accumulated_samples);
      // No host query work ran during this report, so accumulated_samples is 0.
      // Reuse the previous cached delta instead of writing 0. Guests often
      // drive culling decisions off the last visible result.
      if (final_value == 0 && logical.cached_delta != 0) {
        cached_delta = logical.cached_delta;
      } else {
        cached_delta = final_value;
      }
      logical.cached_delta = cached_delta;
      fast_zpd_report_cached_values_[report_record_base] = {
          cached_delta, logical.slot_sequence_id};
      final_value = cached_delta;
    } else if (logical.cached_delta != 0) {
      cached_delta = logical.cached_delta;
    }
  }

  if (missing_logical_report) {
    active_host_zpd_query_segment_ = {};
    return false;
  }

  zpd_report_controller_->QueueGuestReportWrite(report_record_base,
                                                report_handle);
  if (resolved_immediately) {
    zpd_report_controller_->SetReportResolved(report_handle, final_value);
    zpd_report_controller_->RetireResolvedReports();
  }

  if (cvars::occlusion_query_log) {
    XELOGI(
        "ZPD: EndGuestZPDReport report_address=0x{:08X} begin_record=0x{:08X} "
        "end_record=0x{:08X} forced={} resolved_immediately={} "
        "pending_segments={} final_value={} cached_delta={}",
        report_address, begin_record, report_record_base, guest_forced_end,
        resolved_immediately, pending_segments, final_value, cached_delta);
  }

  // For batched query patterns the END event fires at a different address than
  // the slot's own END record (stored at BEGIN time). The game pre-stamps both
  // with sentinels and polls both. Write begin_value to the stored END record
  // immediately so the "before" snapshot sentinel clears in both modes.
  bool has_cross_slot_end =
      stored_end_record && stored_end_record != report_record_base;

  if (IsFastZPDPathEnabled()) {
    bool write_begin_record = begin_record && report_record_base &&
                              begin_record != report_record_base;
    CommitGuestZPDReportDataWithResolvedBeginValue(
        begin_record, report_record_base, begin_value, cached_delta,
        write_begin_record);
    if (has_cross_slot_end) {
      if (cvars::occlusion_query_log) {
        XELOGI(
            "ZPD: EndGuestZPDReport cross-slot before-snapshot write "
            "record=0x{:08X} begin_value={}",
            stored_end_record, begin_value);
      }
      CommitGuestZPDReportDataWithResolvedBeginValue(0, stored_end_record, 0,
                                                     begin_value, false);
    }
  } else if (!resolved_immediately) {
    TryPumpZPDQueryResolves();
    if (zpd_report_controller_->HasQueuedWriteForAddress(report_record_base)) {
      pending_strict_zpd_retire_handle_ = report_handle;
      pending_strict_zpd_retire_stall_count_ = 0;
    }

    if (has_cross_slot_end) {
      if (cvars::occlusion_query_log) {
        XELOGI(
            "ZPD: EndGuestZPDReport cross-slot before-snapshot write "
            "record=0x{:08X} begin_value={}",
            stored_end_record, begin_value);
      }
      CommitGuestZPDReportDataWithResolvedBeginValue(0, stored_end_record, 0,
                                                     begin_value, false);
    }

    // Write a speculative result now so the sentinel is gone before the next
    // frame. The real result will overwrite this once it resolves.
    if (IsStrictImmediateSentinelClearEnabled()) {
      bool write_begin_record = begin_record && report_record_base &&
                                begin_record != report_record_base;
      uint32_t speculative = cached_delta;
      if (write_begin_record && speculative == 0) {
        speculative =
            static_cast<uint32_t>(cvars::occlusion_query_fast_cached_delta);
      }
      if (cvars::occlusion_query_log) {
        XELOGI(
            "ZPD: EndGuestZPDReport strict sentinel clear "
            "record=0x{:08X} speculative={} begin_value={}",
            report_record_base, speculative, begin_value);
      }
      CommitGuestZPDReportDataWithResolvedBeginValue(
          begin_record, report_record_base, begin_value, speculative,
          write_begin_record);
    }
  }

  guest_zpd_report_stats_.logical_ended++;
  active_host_zpd_query_segment_ = {};
  return true;
}

void CommandProcessor::ResumeActiveHostZPDQuerySegment(
    bool can_close_submission) {
  if (!cvars::occlusion_query_enable ||
      !active_host_zpd_query_segment_.logical_active ||
      active_host_zpd_query_segment_.segment_active ||
      !active_host_zpd_query_segment_.segment_pending_begin ||
      !CanOpenHostZPDQueryNow()) {
    return;
  }

  EnsureZPDHostQueryResources();
  if (!IsHostZPDQueryPoolReady()) {
    guest_zpd_report_stats_.failed++;
    return;
  }

  TryPumpZPDQueryResolves();

  uint32_t query_index = UINT32_MAX;
  uint32_t query_generation = 0;
  HostZPDQueryOpenResult open_result =
      OpenHostZPDQuery(query_index, query_generation, can_close_submission);
  switch (open_result) {
    case HostZPDQueryOpenResult::kOpened:
      break;
    case HostZPDQueryOpenResult::kDeferred:
      return;
    case HostZPDQueryOpenResult::kPoolExhausted: {
      guest_zpd_report_stats_.pool_exhausted++;
      // Pool is full and we're in fast mode. Inject a nonzero sample count so
      // the report resolves without stalling for a free slot.
      if (IsFastZPDPathEnabled()) {
        XenosReportController::ReportHandle report_handle =
            active_host_zpd_query_segment_.report_handle;
        auto it = logical_zpd_reports_.find(report_handle);
        if (it != logical_zpd_reports_.end()) {
          it->second.accumulated_samples = std::max<uint64_t>(
              it->second.accumulated_samples,
              static_cast<uint64_t>(cvars::occlusion_query_fast_cached_delta));
        }
        active_host_zpd_query_segment_.segment_pending_begin = false;
        return;
      }
      guest_zpd_report_stats_.failed++;
      return;
    }
    case HostZPDQueryOpenResult::kFailed:
    default:
      guest_zpd_report_stats_.failed++;
      return;
  }

  active_host_zpd_query_segment_.query_index = query_index;
  active_host_zpd_query_segment_.query_generation = query_generation;
  active_host_zpd_query_segment_.segment_active = true;
  active_host_zpd_query_segment_.segment_pending_begin = false;
  guest_zpd_report_stats_.segments_begun++;
}

// Closes the current host query segment and queues it for resolve. A logical
// report can span multiple segments if split by submissions or render passes.
// Results are accumulated before the final write.
void CommandProcessor::SplitActiveHostZPDQuerySegment() {
  if (!cvars::occlusion_query_enable ||
      !active_host_zpd_query_segment_.segment_active) {
    return;
  }

  uint64_t submission = 0;
  uint32_t query_index = active_host_zpd_query_segment_.query_index;
  uint32_t query_generation = active_host_zpd_query_segment_.query_generation;
  if (!CloseHostZPDQuery(query_index, query_generation, submission)) {
    active_host_zpd_query_segment_.segment_active = false;
    active_host_zpd_query_segment_.segment_pending_begin =
        active_host_zpd_query_segment_.logical_active;
    active_host_zpd_query_segment_.query_index = UINT32_MAX;
    active_host_zpd_query_segment_.query_generation = 0;
    guest_zpd_report_stats_.failed++;
    return;
  }

  PendingHostZPDQueryResolve resolve;
  resolve.submission = submission;
  resolve.query_index = query_index;
  resolve.query_generation = query_generation;
  resolve.report_handle = active_host_zpd_query_segment_.report_handle;
  host_zpd_query_resolves_in_flight_.push_back(resolve);

  auto it = logical_zpd_reports_.find(resolve.report_handle);
  if (it != logical_zpd_reports_.end()) {
    it->second.pending_segments++;
    it->second.last_segment_end_submission = resolve.submission;
  }

  active_host_zpd_query_segment_.segment_active = false;
  active_host_zpd_query_segment_.segment_pending_begin =
      active_host_zpd_query_segment_.logical_active;
  active_host_zpd_query_segment_.query_index = UINT32_MAX;
  active_host_zpd_query_segment_.query_generation = 0;
  guest_zpd_report_stats_.segments_ended++;
}

void CommandProcessor::ProcessCompletedHostZPDQueryResolves(
    uint64_t completed_submission) {
  if (!cvars::occlusion_query_enable || !zpd_report_controller_) {
    return;
  }

  struct CompletedResolve {
    XenosReportController::ReportHandle report_handle =
        XenosReportController::kInvalidReportHandle;
    uint64_t raw_samples = 0;
  };
  struct ResolvedReport {
    XenosReportController::ReportHandle report_handle =
        XenosReportController::kInvalidReportHandle;
    uint32_t delta_value = 0;
  };

  std::vector<PendingHostZPDQueryResolve> ready_resolves;
  std::vector<CompletedResolve> completed_resolves;
  std::vector<ResolvedReport> resolved_reports;
  uint64_t discarded_stale = 0;

  PrepareHostZPDReadback(completed_submission);

  while (!host_zpd_query_resolves_in_flight_.empty()) {
    PendingHostZPDQueryResolve resolve =
        host_zpd_query_resolves_in_flight_.front();
    if (resolve.submission > completed_submission) {
      break;
    }
    host_zpd_query_resolves_in_flight_.pop_front();
    ready_resolves.push_back(resolve);
  }

  // Read and release all ready query slots first, then resolve reports. The
  // split lets released slots feed back into the pool before we check whether
  // the logical report can retire.
  completed_resolves.reserve(ready_resolves.size());
  for (const PendingHostZPDQueryResolve& resolve : ready_resolves) {
    uint64_t raw_samples = GetHostZPDQueryResult(resolve.query_index);
    bool is_valid = IsHostZPDQueryResultValid(resolve.query_index,
                                              resolve.query_generation);
    if (!is_valid) {
      ++discarded_stale;
      continue;
    }

    ReleaseHostZPDQuery(resolve.query_index, resolve.query_generation);
    completed_resolves.push_back({resolve.report_handle, raw_samples});
  }

  guest_zpd_report_stats_.resolves_discarded_stale += discarded_stale;

  for (const CompletedResolve& resolve : completed_resolves) {
    auto it = logical_zpd_reports_.find(resolve.report_handle);
    if (it == logical_zpd_reports_.end()) {
      guest_zpd_report_stats_.resolves_discarded_no_logical++;
      continue;
    }

    LogicalGuestZPDReport& logical = it->second;
    if (logical.pending_segments) {
      logical.pending_segments--;
    }
    logical.accumulated_samples += resolve.raw_samples;

    if (logical.ended && logical.pending_segments == 0) {
      uint32_t final_value =
          NormalizeZPDReportSampleCount(logical.accumulated_samples);
      logical.cached_delta = final_value;
      if (logical.end_record) {
        fast_zpd_report_cached_values_[logical.end_record] = {
            final_value, logical.slot_sequence_id};
      }
      resolved_reports.push_back({resolve.report_handle, final_value});
      guest_zpd_report_stats_.resolves_completed++;
    }
  }

  for (const ResolvedReport& resolved_report : resolved_reports) {
    zpd_report_controller_->SetReportResolved(resolved_report.report_handle,
                                              resolved_report.delta_value);
  }
  if (!resolved_reports.empty()) {
    zpd_report_controller_->RetireResolvedReports();
  }
}

bool CommandProcessor::IsFastZPDPathEnabled() const {
  return cvars::occlusion_query_fast;
}

bool CommandProcessor::IsStrictImmediateSentinelClearEnabled() const {
  return !IsFastZPDPathEnabled() &&
         cvars::occlusion_query_strict_immediate_sentinel_clear;
}

void CommandProcessor::TryPumpZPDQueryResolves() {
  if (!cvars::occlusion_query_enable || !zpd_report_controller_) {
    return;
  }

  uint64_t completed_submission = GetHostZPDCompletedSubmission();
  if (completed_submission == 0) {
    return;
  }

  ProcessCompletedHostZPDQueryResolves(completed_submission);
}

bool CommandProcessor::AwaitAndPumpZPDQueryResolves(
    XenosReportController::ReportHandle report_handle) {
  if (!cvars::occlusion_query_enable || !zpd_report_controller_) {
    return false;
  }

  auto it = logical_zpd_reports_.find(report_handle);
  if (it == logical_zpd_reports_.end()) {
    return false;
  }

  uint64_t wait_for_submission = it->second.last_segment_end_submission;
  if (wait_for_submission == 0) {
    if (it->second.pending_segments == 0 && it->second.ended) {
      zpd_report_controller_->RetireResolvedReports();
      return true;
    }
    return false;
  }

  if (cvars::occlusion_query_log) {
    XELOGI(
        "ZPD: AwaitAndPumpZPDQueryResolves handle={} "
        "wait_submission={} completed_submission={}",
        report_handle, wait_for_submission, GetHostZPDCompletedSubmission());
  }

  uint64_t current_submission = GetHostZPDCurrentSubmission();
  // If the submission we need hasn't been sent to the GPU yet, close it first.
  // Waiting on an unfired fence would block forever.
  if (current_submission != 0 &&
      wait_for_submission >= current_submission &&
      CanEndHostZPDSubmissionImmediately()) {
    PrepareToWaitForHostZPDSubmission();
    EndHostZPDSubmission(false);
  }

  uint64_t completed_before = GetHostZPDCompletedSubmission();
  // Only await if the submission has actually been signaled.
  current_submission = GetHostZPDCurrentSubmission();
  if (wait_for_submission > completed_before &&
      wait_for_submission < current_submission) {
    AwaitHostZPDSubmissionAndUpdateCompleted(wait_for_submission);
    uint64_t completed_after = GetHostZPDCompletedSubmission();
    if (completed_after > completed_before) {
      ProcessCompletedHostZPDQueryResolves(completed_after);
    }
  }

  it = logical_zpd_reports_.find(report_handle);
  if (it == logical_zpd_reports_.end()) {
    return true;
  }
  return it->second.pending_segments == 0 && it->second.ended;
}

void CommandProcessor::MaybeAwaitStrictZPDReportRetirement() {
  if (IsFastZPDPathEnabled() || !zpd_report_controller_ ||
      pending_strict_zpd_retire_handle_ ==
          XenosReportController::kInvalidReportHandle) {
    return;
  }

  XenosReportController::ReportHandle handle_to_await =
      pending_strict_zpd_retire_handle_;

  if (cvars::occlusion_query_log) {
    XELOGI("ZPD: MaybeAwaitStrictZPDReportRetirement handle={}",
           handle_to_await);
  }

  if (AwaitAndPumpZPDQueryResolves(handle_to_await)) {
    pending_strict_zpd_retire_handle_ =
        XenosReportController::kInvalidReportHandle;
    pending_strict_zpd_retire_stall_count_ = 0;
    return;
  }

  auto logical_report = logical_zpd_reports_.find(handle_to_await);
  if (logical_report == logical_zpd_reports_.end()) {
    // If the report is already gone it retired through another path.
    // Clear so we don't spin on a handle that no longer exists.
    pending_strict_zpd_retire_handle_ =
        XenosReportController::kInvalidReportHandle;
    pending_strict_zpd_retire_stall_count_ = 0;
    return;
  }

  uint32_t max_stalls =
      cvars::occlusion_query_strict_retire_max_stalls > 0
          ? static_cast<uint32_t>(
                cvars::occlusion_query_strict_retire_max_stalls)
          : 16u;
  if (++pending_strict_zpd_retire_stall_count_ >= max_stalls) {
    if (cvars::occlusion_query_log) {
      XELOGI(
          "ZPD: MaybeAwaitStrictZPDReportRetirement stall limit reached "
          "handle={}, abandoning",
          handle_to_await);
    }
    logical_zpd_reports_.erase(logical_report);
    pending_strict_zpd_retire_handle_ =
        XenosReportController::kInvalidReportHandle;
    pending_strict_zpd_retire_stall_count_ = 0;
  }
}

void CommandProcessor::CommitGuestZPDReportDataWithGuestBeginValue(
    uint32_t begin_record, uint32_t report_record_base, uint32_t delta_value,
    bool write_begin_record) {
  if (!report_record_base) {
    return;
  }

  xenos::xe_gpu_depth_sample_counts* begin =
      begin_record
          ? memory_->TranslatePhysical<xenos::xe_gpu_depth_sample_counts*>(
                begin_record)
          : nullptr;
  xenos::xe_gpu_depth_sample_counts* end_report =
      memory_->TranslatePhysical<xenos::xe_gpu_depth_sample_counts*>(
          report_record_base);

  XenosZPDReport::WriteGuestReportDelta(begin, end_report, delta_value,
                                        write_begin_record);
}

// Used only for orphan END writes where no controller BEGIN value is
// available. Reads begin from guest memory, which may reflect a guest write
// rather than our snapshot.
void CommandProcessor::CommitGuestZPDReportDataWithResolvedBeginValue(
    uint32_t begin_record, uint32_t report_record_base, uint32_t begin_value,
    uint32_t delta_value, bool write_begin_record) {
  if (!report_record_base) {
    return;
  }

  xenos::xe_gpu_depth_sample_counts* begin =
      begin_record
          ? memory_->TranslatePhysical<xenos::xe_gpu_depth_sample_counts*>(
                begin_record)
          : nullptr;
  xenos::xe_gpu_depth_sample_counts* end_report =
      memory_->TranslatePhysical<xenos::xe_gpu_depth_sample_counts*>(
          report_record_base);

  XenosZPDReport::WriteGuestReportDeltaWithBeginValue(
      begin, end_report, begin_value, delta_value, write_begin_record);
}

void CommandProcessor::CommitGuestZPDReportCallback(
    XenosReportController::ReportHandle report_handle,
    uint32_t report_record_base, uint32_t begin_value, uint32_t delta_value,
    bool write_begin_report, void* callback_context) {
  CommandProcessor* processor =
      reinterpret_cast<CommandProcessor*>(callback_context);

  uint32_t target_record_base =
      XenosZPDReport::GetRecordBase(report_record_base);
  if (!target_record_base) {
    return;
  }

  // Controller always passes end_record_base (slot base) here, so
  // GetBeginRecordBase gives the right fallback.
  uint32_t begin_record =
      XenosZPDReport::GetBeginRecordBase(target_record_base);
  auto existing_report = processor->logical_zpd_reports_.find(report_handle);
  if (existing_report != processor->logical_zpd_reports_.end()) {
    begin_record = existing_report->second.begin_record;
    processor->logical_zpd_reports_.erase(existing_report);
  } else if (cvars::occlusion_query_log) {
    XELOGI(
        "ZPD: CommitGuestZPDReportCallback missing logical report "
        "handle={} record=0x{:08X}",
        report_handle, target_record_base);
  }

  processor->CommitGuestZPDReportDataWithResolvedBeginValue(
      begin_record, target_record_base, begin_value, delta_value,
      write_begin_report);
}

uint32_t CommandProcessor::NormalizeZPDReportSampleCount(
    uint64_t samples) const {
  if (samples == 0) {
    return 0;
  }

  uint64_t scale_x = GetZPDReportDrawResolutionScaleX();
  uint64_t scale_y = GetZPDReportDrawResolutionScaleY();
  uint64_t scale = scale_x * scale_y;
  // A result upscaling inflated to 3 samples at 2x should resolve to 1, not 0.
  uint64_t normalized = scale <= 1 ? samples : (samples + (scale >> 1)) / scale;
  return static_cast<uint32_t>(std::min<uint64_t>(normalized, UINT32_MAX));
}

#define COMMAND_PROCESSOR CommandProcessor
#include "pm4_command_processor_implement.h"
}  // namespace gpu
}  // namespace xe
