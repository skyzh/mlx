// Copyright © 2024 Apple Inc.

#include "mlx/backend/common/hadamard.h"
#include "mlx/backend/common/compiled.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/metal/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/jit/includes.h"
#include "mlx/backend/metal/kernels.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"

namespace mlx::core {

constexpr int MAX_HADAMARD_THREADS_PER_GROUP = 256;
constexpr int MAX_HADAMARD_BYTES = 32768; // 32KB

std::string gen_hadamard_codelet(int m) {
  // Generate a O(m^2) hadamard codelet for a given M
  // using the hadamard matrices above
  //
  // e.g. m = 2
  // METAL_FUNC void hadamard_m(thread float *x) {
  //   float tmp[2];
  //   tmp[0] = + x[0] + x[1];
  //   tmp[1] = + x[0] - x[1];
  //   for (int i = 0; i < 2; i++) { x[i] = tmp[i]; }
  // }
  //
  auto h_matrices = hadamard_matrices();
  auto& matrix = h_matrices[m];

  std::ostringstream source;
  source << "METAL_FUNC void hadamard_radix_m(thread float *x) {" << std::endl;
  if (m == 1) {
    source << "}" << std::endl;
    return source.str();
  }
  source << "  float tmp[" << m << "];" << std::endl;
  auto start = 1;
  auto end = matrix.find('\n', start);

  int index = 0;
  while (end != std::string_view::npos) {
    source << "  tmp[" << index << "] = ";
    auto row = matrix.substr(start, end - start);
    for (int i = 0; i < row.length(); i++) {
      source << " " << row[i] << " x[" << i << "]";
    }
    source << ";" << std::endl;
    start = end + 1;
    end = matrix.find('\n', start);
    index++;
  }
  source << "  for (int i = 0; i < " << m << "; i++) { x[i] = tmp[i]; }"
         << std::endl;
  source << "}" << std::endl;
  return source.str();
}

void hadamard_inplace(
    array& x,
    int m,
    int n,
    float scale,
    metal::Device& d,
    const Stream& s) {
  int read_width_n = n == 2 ? 2 : 4;
  int read_width_m = (n == 2 || m == 28) ? 2 : 4;
  int max_radix = std::min(n, 16);
  float scale_n = m == 1 ? scale : 1.0;
  float scale_m = scale;

  MTL::Size group_dims_n(n / max_radix, 1, 1);
  MTL::Size grid_dims_n(n / max_radix, x.size() / n, 1);
  MTL::Size group_dims_m(
      std::min(n / read_width_m, MAX_HADAMARD_THREADS_PER_GROUP), 1, 1);
  MTL::Size grid_dims_m(
      group_dims_m.width, x.size() / m / read_width_m / group_dims_m.width, 1);

  // Make the kernel
  std::string kname;
  kname.reserve(32);
  concatenate(kname, "hadamard_", n * m, "_", type_to_name(x));
  auto lib = d.get_library(kname, [&]() {
    std::string kernel;
    concatenate(
        kernel,
        metal::utils(),
        gen_hadamard_codelet(m),
        metal::hadamard(),
        get_template_definition(
            "n" + kname,
            "hadamard_n",
            get_type_string(x.dtype()),
            n,
            max_radix,
            read_width_n),
        get_template_definition(
            "m" + kname,
            "hadamard_m",
            get_type_string(x.dtype()),
            n,
            m,
            read_width_m));
    return kernel;
  });

  // Launch the transform for n
  auto& compute_encoder = d.get_command_encoder(s.index);
  auto kernel = d.get_kernel("n" + kname, lib);
  compute_encoder.set_compute_pipeline_state(kernel);
  compute_encoder.set_input_array(x, 0);
  compute_encoder.set_output_array(x, 1);
  compute_encoder.set_bytes(scale_n, 2);
  compute_encoder.dispatch_threads(grid_dims_n, group_dims_n);

  if (m > 1) {
    auto kernel = d.get_kernel("m" + kname, lib);
    compute_encoder.set_compute_pipeline_state(kernel);
    compute_encoder.set_input_array(x, 0);
    compute_encoder.set_output_array(x, 1);
    compute_encoder.set_bytes(scale_m, 2);
    compute_encoder.dispatch_threads(grid_dims_m, group_dims_m);
  }
}

void Hadamard::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = stream();
  auto& d = metal::device(s.device);
  auto& in = inputs[0];

  // Route to the correct hadamard transform, basically we have
  // 1. n = m * 2^k for m in (1, 12, 20, 28) and 2^k * sizeof(T) <= 32KB
  // 2. n = m * 2^k for m in (1, 12, 20, 28) and 2^k * sizeof(T) > 32KB
  // 3. m from above not in these values (will fail in decompose)
  auto [n, m] = decompose_hadamard(in.shape().back());

  // Case 1
  if (n * in.itemsize() <= MAX_HADAMARD_BYTES) {
    copy_gpu(
        in,
        out,
        in.flags().row_contiguous ? CopyType::Vector : CopyType::General,
        s);
    hadamard_inplace(out, m, n, scale_, d, s);
    return;
  }
}

} // namespace mlx::core
